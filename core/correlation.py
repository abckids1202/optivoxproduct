"""Correlation boundary for turning model outputs into operational entity state.

The vision models remain responsible for observations. This module is the
authoritative in-memory reducer for identity, liveness, presence, and
attendance eligibility decisions consumed by the persistence bridge.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .entities import EntityState, EntityStateStore, normalize_identity_state
from .event_envelope import EventEnvelope, attach_envelope_metadata
from .observations import Observation, ObservationHistory, ObservationType


def _center(bbox):
    if not bbox or len(bbox) < 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) * 0.5,
            (float(bbox[1]) + float(bbox[3])) * 0.5)


class CorrelationCore:
    """Normalizes frame outputs into bounded observations and entity state."""

    def __init__(self, max_entities: int = 128, max_observations_per_type: int = 12,
                 close_after_sec: float = 10.0, ttl_overrides_ms: Optional[Dict[str, int]] = None,
                 identity_stale_after_sec: float = 3.0,
                 face_visible_after_sec: float = 1.0,
                 quality_valid_after_sec: float = 1.5,
                 liveness_valid_after_sec: float = 2.0,
                 identity_confirmation_observations: int = 1,
                 identity_confirmation_window: int = 5,
                 track_switch_distance: float = 250.0,
                 site_id: Optional[str] = None,
                 organization_id: Optional[str] = None,
                 device_id: Optional[str] = None,
                 policy_version: str = "1",
                 policy_valid: bool = True,
                 policy_issues: Optional[Iterable[str]] = None):
        self.site_id = str(site_id or "")
        self.organization_id = str(organization_id or "")
        self.device_id = str(device_id or "")
        self.policy_version = str(policy_version or "1")
        self.policy_valid = bool(policy_valid)
        self.policy_issues = tuple(str(issue) for issue in (policy_issues or ()) if str(issue))
        self.entities = EntityStateStore(
            max_entities=max_entities,
            max_observations_per_type=max_observations_per_type,
            close_after_sec=close_after_sec,
            identity_stale_after_sec=identity_stale_after_sec,
            face_visible_after_sec=face_visible_after_sec,
            quality_valid_after_sec=quality_valid_after_sec,
            liveness_valid_after_sec=liveness_valid_after_sec,
            identity_confirmation_observations=identity_confirmation_observations,
            identity_confirmation_window=identity_confirmation_window,
            track_switch_distance=track_switch_distance,
        )
        self.global_history = ObservationHistory(
            max_per_type=max_observations_per_type,
            max_entities=max_entities,
        )
        self.ttl_overrides_ms = {
            ObservationType(key): max(1, int(value))
            for key, value in (ttl_overrides_ms or {}).items()
            if str(key) in {item.value for item in ObservationType}
        }
        self.frames = 0
        self.observations_created = 0
        self.global_observations = 0
        self.last_frame_id: Optional[int] = None
        self.last_update_ms = 0.0
        self._expired_observations_rejected = 0
        self.identity_conflicts_total = 0
        self.last_identity_conflicts: List[Dict[str, object]] = []
        # None means the database roster has not been connected yet. An
        # empty set is intentional and fails closed for a configured roster.
        self._active_roster: Optional[set[str]] = None

    def set_active_roster(self, names: Optional[Iterable[str]]) -> None:
        """Connect the reducer to the active database roster.

        A model label can be useful for display without being authorized for
        attendance or named security attribution. Keeping this set at the
        correlation boundary prevents callers from reducing roster policy to
        a permissive boolean flag.
        """
        if names is None:
            self._active_roster = None
            return
        self._active_roster = {
            str(name).strip().casefold()
            for name in names
            if str(name).strip()
        }

    @property
    def active_roster_configured(self) -> bool:
        return self._active_roster is not None

    def roster_validation(self, name: Optional[str]) -> Dict[str, object]:
        """Return explicit roster provenance for a canonical identity."""
        normalized = str(name or "").strip().casefold()
        if self._active_roster is None:
            return {"match": True, "configured": False, "status": "NOT_CONFIGURED"}
        matched = bool(normalized and normalized in self._active_roster)
        return {
            "match": matched,
            "configured": True,
            "status": "ACTIVE" if matched else "INACTIVE_OR_MISSING",
        }

    def _add(self, observation: Observation) -> Observation:
        override = self.ttl_overrides_ms.get(observation.observation_type)
        if override is not None:
            observation.ttl_ms = override
            observation.expires_at_monotonic = observation.observed_at_monotonic + override / 1000.0
        self.observations_created += 1
        if observation.entity_track_id is None:
            self.global_observations += 1
            self.global_history.add(observation)
        else:
            self.entities.attach(observation)
        return observation

    def _nearest_track(self, bbox, tracked: Dict[int, Tuple[int, int]]) -> Optional[int]:
        center = _center(bbox)
        if center is None:
            return None
        candidates = []
        for track_id, track_center in tracked.items():
            distance = math.hypot(center[0] - track_center[0], center[1] - track_center[1])
            candidates.append((distance, int(track_id)))
        if not candidates:
            return None
        distance, track_id = min(candidates)
        return track_id if distance <= 150.0 else None

    def _correlate_security_events(
        self,
        security_events: Optional[Iterable[tuple]],
        source_frame_id: Optional[int],
        camera_id: str,
        observed_at_monotonic: float,
        observed_at_wallclock: Optional[str],
    ) -> Dict[str, list]:
        """Attach security observations to the current canonical entities.

        Security rules may run after the identity reducer has established the
        current entity state. This method deliberately does not advance the
        frame reducer or create a second person observation; it only records
        the already-correlated security decisions that are safe to persist.
        """
        correlated_event_tuples = []
        correlated_event_records = []
        for event in security_events or ():
            if not event:
                continue
            event_type = str(event[0])
            metadata = dict(event[4]) if len(event) > 4 and isinstance(event[4], dict) else {}
            metadata.setdefault("policy_version", self.policy_version)
            track_id = metadata.get("track_id")
            try:
                track_id = int(track_id) if track_id is not None else None
            except (TypeError, ValueError):
                track_id = None
            source_entity_type = str(metadata.get("entity_type") or "person").casefold()
            # Vehicle tracks use a separate namespace from person tracks. A
            # numeric ID collision must never make a vehicle event inherit a
            # person's identity or attendance context.
            entity = (
                self.entities.get(track_id)
                if track_id is not None and source_entity_type == "person"
                else None
            )
            if entity is not None:
                entity.refresh(observed_at_monotonic)
                identity = entity.identity
                roster = self.roster_validation(identity.confirmed_name)
                metadata.update({
                    "entity_id": entity.entity_id,
                    "track_generation": entity.track_generation,
                    "lifecycle_state": entity.lifecycle_state.value,
                    "identity_state": identity.state,
                    "confirmed_name": identity.confirmed_name,
                    "identity_evidence_fresh": identity.current_evidence_fresh,
                    "liveness_state": entity.liveness.state,
                    "attendance_eligibility": bool(
                        entity.attendance_eligibility and roster["match"]
                    ),
                    "roster_match": bool(identity.confirmed_name and roster["match"]),
                    "roster_validation": roster,
                    # The reducer runs before the persistence worker has an
                    # integer presence-session row. Keep a stable context key
                    # for correlation, but never present the entity key as a
                    # persisted session identifier.
                    "presence_session_key": (
                        f"{entity.entity_id}:generation:{entity.track_generation}"
                    ),
                    "presence_session_id": metadata.get("presence_session_id"),
                    "source_frame_id": source_frame_id,
                })
                zone_id = metadata.get("zone_id")
                if zone_id:
                    if event_type == "ZONE_EXIT":
                        entity.current_zone = None
                    else:
                        entity.current_zone = str(zone_id)
            # Keep one stable metadata contract for both person-attributed
            # signals and global/object signals. Consumers should not need to
            # guess whether a missing key means "not correlated" or "not
            # applicable".
            metadata.setdefault("entity_id", None)
            metadata.setdefault("entity_type", source_entity_type)
            metadata.setdefault("track_generation", None)
            metadata.setdefault(
                "lifecycle_state",
                "UNRESOLVED" if source_entity_type == "person" else "NOT_APPLICABLE",
            )
            metadata.setdefault(
                "identity_state",
                "UNRESOLVED" if source_entity_type == "person" else "NOT_APPLICABLE",
            )
            metadata.setdefault("confirmed_name", None)
            metadata.setdefault("identity_evidence_fresh", False)
            metadata.setdefault(
                "liveness_state",
                "NOT_EVALUATED" if source_entity_type == "person" else "NOT_APPLICABLE",
            )
            metadata.setdefault("attendance_eligibility", False)
            metadata.setdefault("roster_match", False)
            metadata.setdefault(
                "roster_validation",
                {
                    "match": False,
                    "configured": self.active_roster_configured,
                    "status": (
                        "NOT_APPLICABLE" if source_entity_type != "person"
                        else "NOT_CONFIGURED" if not self.active_roster_configured
                        else "INACTIVE_OR_MISSING"
                    ),
                },
            )
            metadata.setdefault("presence_session_key", None)
            metadata.setdefault("camera_id", camera_id)
            if self.site_id:
                metadata.setdefault("site_id", self.site_id)
            if self.organization_id:
                metadata.setdefault("organization_id", self.organization_id)
            if self.device_id:
                metadata.setdefault("device_id", self.device_id)
            metadata.setdefault("source_frame_id", source_frame_id)
            target = str(event[1]) if len(event) > 1 else "SYSTEM"
            # Never let a raw ID_* target become a named person unless the
            # current entity is confirmed by the correlation reducer.
            if entity is not None:
                identity = entity.identity
                if (identity.state == "CONFIRMED" and identity.confirmed_name
                        and identity.current_evidence_fresh
                        and metadata.get("roster_match", False)):
                    if target.startswith("ID_") or target in {"PERSON", "UNKNOWN"}:
                        target = identity.confirmed_name
                elif target == "PERSON" or target.startswith("ID_") or target not in {"SYSTEM"}:
                    target = "UNKNOWN"
            elif target == "PERSON" or target.startswith("ID_"):
                # A raw tracker label without a live canonical entity has no
                # attributable identity. Keep the security signal reviewable,
                # but do not leak a stale numeric identity downstream.
                target = "UNKNOWN"
            confidence = float(event[2]) if len(event) > 2 else 0.0
            details = event[3] if len(event) > 3 else ""
            envelope = EventEnvelope.from_security_tuple(
                (event_type, target, confidence, details, metadata),
                camera_id=camera_id,
                occurred_at=observed_at_wallclock or "",
                source_frame_id=source_frame_id,
                entity_id=metadata.get("entity_id"),
                # A real database session ID is attached by the operations
                # worker after the session upsert. The correlation reducer
                # must not fabricate one from an entity ID.
                presence_session_id=metadata.get("presence_session_id"),
                metadata=metadata,
            )
            metadata = attach_envelope_metadata(metadata, envelope)
            correlated_event_tuples.append((event_type, target, confidence, details, metadata))
            correlated_event_records.append({
                "event_type": event_type,
                "target": target,
                "confidence": confidence,
                "details": details,
                "metadata": metadata,
                "envelope": envelope.to_dict(),
            })
            self._add(Observation(
                ObservationType.SECURITY_SIGNAL,
                camera_id=camera_id,
                source_frame_id=source_frame_id,
                observed_at_monotonic=observed_at_monotonic,
                observed_at_wallclock=observed_at_wallclock or "",
                producer="SecuritySignalEngine",
                entity_track_id=track_id if source_entity_type == "person" else None,
                confidence=confidence,
                confidence_type="security_signal_confidence",
                value=event_type,
                metadata={**metadata, "event_type": event_type, "details": details,
                          "entity_type": source_entity_type},
            ))
            if entity is not None and metadata.get("zone_id"):
                self._add(Observation(
                    ObservationType.ZONE_MEMBERSHIP,
                    camera_id=camera_id,
                    source_frame_id=source_frame_id,
                    observed_at_monotonic=observed_at_monotonic,
                    observed_at_wallclock=observed_at_wallclock or "",
                    producer="SecuritySignalEngine",
                    entity_track_id=track_id,
                    value=None if event_type == "ZONE_EXIT" else metadata.get("zone_id"),
                    metadata={"zone_name": metadata.get("zone_name"), "event_type": event_type},
                ))
        return {
            "security_event_tuples": correlated_event_tuples,
            "security_events": correlated_event_records,
        }

    def update(
        self,
        tracked: Dict[int, Tuple[int, int]],
        faces_info: Optional[Iterable[dict]] = None,
        object_detections: Optional[Iterable[dict]] = None,
        pose_result: Optional[dict] = None,
        source_frame_id: Optional[int] = None,
        camera_id: str = "cam_0",
        observed_at_monotonic: Optional[float] = None,
        observed_at_wallclock: Optional[str] = None,
        provenance: Optional[Dict[str, Dict[str, object]]] = None,
        security_events: Optional[Iterable[tuple]] = None,
    ) -> Dict[str, object]:
        started = time.perf_counter()
        now = time.monotonic() if observed_at_monotonic is None else observed_at_monotonic
        entities = self.entities.update_tracks(tracked, camera_id=camera_id, now=now)
        presence_transitions = self.entities.consume_transitions()
        self.frames += 1
        self.last_frame_id = source_frame_id
        provenance = provenance or {}

        def source_for(name: str):
            source = provenance.get(name, {}) or {}
            return (
                source.get("frame_id") if source.get("frame_id") is not None else source_frame_id,
                source.get("monotonic") if source.get("monotonic") is not None else now,
                source.get("wallclock") or observed_at_wallclock or "",
            )

        face_frame_id, face_observed_at, face_wallclock = source_for("faces")
        object_frame_id, object_observed_at, object_wallclock = source_for("objects")
        pose_frame_id, pose_observed_at, pose_wallclock = source_for("pose")

        for track_id, center in tracked.items():
            entity = self.entities.get(track_id)
            velocity = entity.motion.velocity if entity else (0.0, 0.0)
            self._add(Observation(
                ObservationType.PERSON_DETECTED,
                camera_id=camera_id,
                source_frame_id=source_frame_id,
                observed_at_monotonic=now,
                observed_at_wallclock=observed_at_wallclock or "",
                producer="CentroidTracker",
                entity_track_id=int(track_id),
                confidence_type="tracking_presence",
                value={"center": tuple(center)},
            ))
            self._add(Observation(
                ObservationType.TRACK_MOTION,
                camera_id=camera_id,
                source_frame_id=source_frame_id,
                observed_at_monotonic=now,
                observed_at_wallclock=observed_at_wallclock or "",
                producer="CentroidTracker",
                entity_track_id=int(track_id),
                confidence_type="tracking_score",
                metadata={"velocity": velocity},
                value={"center": tuple(center)},
            ))

        for face in faces_info or ():
            bbox = face.get("bbox")
            track_id = face.get("oid")
            if not isinstance(track_id, int) or track_id <= 0:
                track_id = self._nearest_track(bbox, tracked)
            self._add(Observation(
                ObservationType.FACE_DETECTED,
                producer="InsightFace",
                model_name="buffalo_l",
                **{
                    "camera_id": camera_id,
                    "source_frame_id": face_frame_id,
                    "observed_at_monotonic": face_observed_at,
                    "observed_at_wallclock": face_wallclock,
                    "entity_track_id": track_id,
                    "bbox": bbox,
                },
                confidence=face.get("det_score"),
                confidence_type="face_detection_score",
                metadata={"track_association": "runtime_oid" if face.get("oid") == track_id else "nearest_centroid"},
            ))
            self._add(Observation(
                ObservationType.FACE_QUALITY,
                producer="ImageQualityScorer",
                camera_id=camera_id,
                source_frame_id=face_frame_id,
                observed_at_monotonic=face_observed_at,
                observed_at_wallclock=face_wallclock,
                entity_track_id=track_id,
                bbox=bbox,
                quality=face.get("quality_score"),
                confidence_type="face_quality_score",
                metadata={"quality_ok": bool(face.get("quality_ok", False)), "metrics": face.get("quality_metrics", {})},
            ))
            if face.get("yaw") is not None:
                self._add(Observation(
                    ObservationType.FACE_HEAD_POSE,
                    producer="HeadPoseEstimator",
                    camera_id=camera_id,
                    source_frame_id=face_frame_id,
                    observed_at_monotonic=face_observed_at,
                    observed_at_wallclock=face_wallclock,
                    entity_track_id=track_id,
                    bbox=bbox,
                    confidence_type="head_pose_estimate",
                    value=face.get("yaw"),
                    metadata={"yaw": face.get("yaw")},
                ))
            self._add(Observation(
                ObservationType.FACE_IDENTITY_RESULT,
                producer="IdentityReducerAdapter",
                camera_id=camera_id,
                source_frame_id=face_frame_id,
                observed_at_monotonic=face_observed_at,
                observed_at_wallclock=face_wallclock,
                entity_track_id=track_id,
                bbox=bbox,
                confidence=face.get("current_observation_similarity", face.get("confidence")),
                confidence_type="cosine_similarity_or_cached_identity",
                value=face.get("name"),
                metadata={
                    "name": face.get("name"),
                    "identity_state": normalize_identity_state(face.get("identity_state")),
                    "cached_identity_confidence": face.get("cached_identity_confidence"),
                    "current_observation_similarity": face.get("current_observation_similarity"),
                    "second_score": face.get("second_score"),
                    "margin": face.get("margin"),
                    "candidate_hits": face.get("candidate_hits"),
                    "stable_frames": face.get("stable_frames"),
                    "last_verified_at": face.get("last_verified_at"),
                    # Cached names are diagnostic continuity only. Prefer the
                    # explicit producer flag and retain the reason fallback for
                    # older adapters that predate the flag.
                    "cached_only": bool(face.get("cached_only")) or str(
                        face.get("reason") or "") in {
                            "stable_track_cache", "cached_identity", "quality_hold",
                            "WAITING_FOR_GOOD_FACE",
                        },
                    "recognition_reason": face.get("reason"),
                },
            ))
            liveness_details = face.get("liveness_details") or {}
            self._add(Observation(
                ObservationType.LIVENESS_RESULT,
                producer="AntiSpoofDetector",
                camera_id=camera_id,
                source_frame_id=face_frame_id,
                observed_at_monotonic=face_observed_at,
                observed_at_wallclock=face_wallclock,
                entity_track_id=track_id,
                bbox=bbox,
                value=face.get("liveness_status", "NOT_EVALUATED"),
                confidence_type="categorical_liveness_state",
                metadata={
                    "attendance_eligible": bool(face.get("attendance_eligible", False)),
                    "challenge_phase": liveness_details.get("phase"),
                    "failure_reason": liveness_details.get("reason"),
                    "challenge_attempts": liveness_details.get("attempt_number"),
                },
            ))

        pose = pose_result or {}
        if pose:
            pose_value = "FALL" if pose.get("is_fallen") else (
                "HANDS_RAISED" if pose.get("hands_raised") else "POSE_DETECTED")
            pose_track = next(iter(tracked)) if len(tracked) == 1 else None
            self._add(Observation(
                ObservationType.POSE_STATE,
                camera_id=camera_id,
                source_frame_id=pose_frame_id,
                observed_at_monotonic=pose_observed_at,
                observed_at_wallclock=pose_wallclock,
                producer="MediaPipePose",
                entity_track_id=pose_track,
                confidence_type="pose_state",
                value=pose_value,
                metadata={
                    "is_fallen": bool(pose.get("is_fallen")),
                    "hands_raised": bool(pose.get("hands_raised")),
                    "association": "single_track" if pose_track is not None else "global_roi",
                },
            ))

        for detection in object_detections or ():
            model_name = str(
                detection.get("model_name") or detection.get("source_model") or "YOLO"
            )
            self._add(Observation(
                ObservationType.OBJECT_DETECTED,
                camera_id=camera_id,
                source_frame_id=object_frame_id,
                observed_at_monotonic=object_observed_at,
                observed_at_wallclock=object_wallclock,
                producer=model_name,
                model_name=model_name,
                model_version=detection.get("model_version"),
                confidence=detection.get("confidence"),
                confidence_type="detector_confidence",
                bbox=detection.get("bbox"),
                value=detection.get("class_name"),
                metadata={
                    "category": detection.get("category"),
                    "event_type": detection.get("event_type", "OBJECT_DETECTED"),
                    "execution_mode": detection.get("execution_mode", "active"),
                    "entity_association": "not_established",
                },
            ))

        # Resolve duplicate live confirmations before security signals are
        # attributed. This keeps attendance and security on the same identity
        # authority when a tracker or face-to-track association duplicates a
        # roster match.
        self.last_identity_conflicts = self.entities.enforce_identity_exclusivity()
        self.identity_conflicts_total += len(self.last_identity_conflicts)

        # Persist identity collisions through the same envelope boundary as
        # security observations. This is an identity-integrity review signal,
        # not a danger classification or an automatic external alert.
        identity_conflict_events = [
            (
                "IDENTITY_CONFLICT",
                "UNKNOWN",
                0.95,
                "Two live entities produced the same confirmed roster identity; "
                "the weaker entity was downgraded.",
                {
                    "track_id": conflict.get("loser_track_id"),
                    "entity_id": conflict.get("loser_entity_id"),
                    "track_generation": conflict.get("loser_track_generation"),
                    "entity_type": "person",
                    "identity": conflict.get("identity"),
                    "winner_entity_id": conflict.get("winner_entity_id"),
                    "winner_track_id": conflict.get("winner_track_id"),
                    "winner_track_generation": conflict.get("winner_track_generation"),
                    "reason": conflict.get("reason"),
                },
            )
            for conflict in self.last_identity_conflicts
        ]

        correlated = self._correlate_security_events(
            security_events, source_frame_id, camera_id, now, observed_at_wallclock)
        correlated_conflicts = self._correlate_security_events(
            identity_conflict_events, source_frame_id, camera_id, now,
            observed_at_wallclock)

        self.entities.history.prune(now)
        self.global_history.prune(now)
        self.last_update_ms = (time.perf_counter() - started) * 1000.0
        return {
            "enabled": True,
            "decision_boundary": "CORRELATION_CORE",
            "frame_id": source_frame_id,
            "camera_id": camera_id,
            "entities": self._entity_snapshot(now),
            "security_events": (
                correlated_conflicts["security_events"]
                + correlated["security_events"]
            ),
            # The runtime bridge consumes these enriched tuples for event,
            # alert, and incident persistence. Raw model events remain useful
            # for diagnostics but are no longer the operational contract.
            "security_event_tuples": (
                correlated_conflicts["security_event_tuples"]
                + correlated["security_event_tuples"]
            ),
            "presence_transitions": presence_transitions,
            "identity_conflicts": list(self.last_identity_conflicts),
            "stats": self.stats(),
        }

    def correlate_security_events(
        self,
        security_events: Optional[Iterable[tuple]],
        source_frame_id: Optional[int] = None,
        camera_id: str = "cam_0",
        observed_at_monotonic: Optional[float] = None,
        observed_at_wallclock: Optional[str] = None,
    ) -> Dict[str, list]:
        """Correlate security output after identity state is canonical.

        This second-stage hook exists to avoid a circular dependency: zone
        policy needs canonical identity, while entity state also needs the
        resulting security observation. It does not increment frame counts or
        re-run recognition, so it is safe for the real-time path.
        """
        now = time.monotonic() if observed_at_monotonic is None else observed_at_monotonic
        result = self._correlate_security_events(
            security_events, source_frame_id, camera_id, now, observed_at_wallclock)
        self.entities.history.prune(now)
        self.global_history.prune(now)
        return result

    def stats(self) -> Dict[str, object]:
        return {
            "frames": self.frames,
            "observations_created": self.observations_created,
            "global_observations": self.global_observations,
            "last_frame_id": self.last_frame_id,
            "last_update_ms": round(self.last_update_ms, 3),
            "expired_observations_rejected": self.entities.expired_observations_rejected,
            "identity_conflicts_total": self.identity_conflicts_total,
            "identity_conflicts_last_update": len(self.last_identity_conflicts),
            "active_roster_configured": self.active_roster_configured,
            "active_roster_count": len(self._active_roster or ()),
            "policy_version": self.policy_version,
            "policy_valid": self.policy_valid,
            "policy_issues": list(self.policy_issues),
            "entities": self.entities.stats(),
            "global_history": self.global_history.stats(),
        }

    def _entity_snapshot(self, now: Optional[float] = None) -> List[Dict[str, object]]:
        """Return entity state with database-roster authorization applied."""
        result = self.entities.snapshot(now)
        for entity in result:
            identity = entity.get("identity") or {}
            validation = self.roster_validation(identity.get("confirmed_name"))
            entity["roster_match"] = bool(
                identity.get("confirmed_name") and validation["match"]
            )
            entity["roster_validation"] = validation
            entity["attendance_eligibility"] = bool(
                entity.get("attendance_eligibility") and entity["roster_match"]
            )
        return result

    def attendance_decision(
        self,
        track_id: int,
        expected_name: Optional[str] = None,
        active_roster: object = True,
    ) -> Dict[str, object]:
        """Return the single authoritative attendance gate for one entity.

        This is intentionally stricter than recognition display. A cached or
        visually plausible name is not enough; the current correlated entity
        must still be visible, quality-valid, live, and confirmed.
        """
        entity = self.entities.get(track_id)
        if entity is None:
            return {
                "eligible": False,
                "reason": "entity_not_active",
                "policy_version": self.policy_version,
                "policy_valid": self.policy_valid,
            }
        gate = entity.attendance_gate()
        identity = entity.identity
        reasons = [] if gate.get("eligible") else str(gate.get("reason", "rejected")).split(",")
        if not self.policy_valid:
            reasons.append("policy_invalid")
        if not entity.attendance_eligibility and "entity_attendance_not_eligible" not in reasons:
            reasons.append("entity_attendance_not_eligible")
        if isinstance(active_roster, bool):
            roster = self.roster_validation(identity.confirmed_name)
            roster_match = bool(
                active_roster and identity.confirmed_name and roster["match"]
            )
        else:
            candidate_roster = {
                str(name).strip().casefold()
                for name in (active_roster or ())
                if str(name).strip()
            }
            normalized_name = str(identity.confirmed_name or "").strip().casefold()
            roster_match = bool(normalized_name and normalized_name in candidate_roster)
            roster = {
                "match": roster_match,
                "configured": True,
                "status": "ACTIVE" if roster_match else "INACTIVE_OR_MISSING",
            }
        if not roster_match:
            reasons.append("person_not_active_in_roster")
        if (expected_name and (not identity.confirmed_name or
                               identity.confirmed_name.casefold() != str(expected_name).casefold())):
            reasons.append("identity_name_mismatch")
        return {
            "eligible": not reasons,
            "reason": "eligible" if not reasons else ",".join(reasons),
            "entity_id": entity.entity_id,
            # A prior confirmed name may remain in the reducer for display
            # continuity, but it is not a current identity decision once the
            # state is contradicted, occluded, expired, or spoof-suspect.
            # Never expose that stale name as an attendance decision.
            "person_name": (
                identity.confirmed_name
                if identity.state == "CONFIRMED" else None
            ),
            "identity_state": identity.state,
            "liveness_state": entity.liveness.state,
            "quality_ok": gate.get("quality_ok"),
            "attendance_eligibility": entity.attendance_eligibility,
            "entity_gate": gate,
            "confidence": identity.best_score,
            "second_score": identity.second_score,
            "margin": identity.margin,
            "confirmation_hits": identity.confirmation_hits,
            "contradiction_count": identity.contradiction_count,
            "identity_evidence_fresh": identity.current_evidence_fresh,
            "roster_match": roster_match,
            "roster_validation": roster,
            "policy_version": self.policy_version,
            "policy_valid": self.policy_valid,
            "policy_issues": list(self.policy_issues),
            "track_generation": entity.track_generation,
            "source_frame_id": entity.last_source_frame_id,
            "observed_at": entity.last_observed_wallclock,
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "last_frame_id": self.last_frame_id,
            "last_update_ms": round(self.last_update_ms, 3),
            "stats": self.stats(),
            "roster": {
                "configured": self.active_roster_configured,
                "active_count": len(self._active_roster or ()),
            },
            "policy_version": self.policy_version,
            "policy_valid": self.policy_valid,
            "policy_issues": list(self.policy_issues),
            "entities": self._entity_snapshot(),
        }
