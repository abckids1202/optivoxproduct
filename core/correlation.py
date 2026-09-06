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
from .observations import Observation, ObservationHistory, ObservationType


def _center(bbox):
    if not bbox or len(bbox) < 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) * 0.5,
            (float(bbox[1]) + float(bbox[3])) * 0.5)


class CorrelationCore:
    """Normalizes frame outputs into bounded observations and entity state."""

    def __init__(self, max_entities: int = 128, max_observations_per_type: int = 12,
                 close_after_sec: float = 10.0, ttl_overrides_ms: Optional[Dict[str, int]] = None):
        self.entities = EntityStateStore(
            max_entities=max_entities,
            max_observations_per_type=max_observations_per_type,
            close_after_sec=close_after_sec,
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
    ) -> Dict[str, object]:
        started = time.perf_counter()
        now = time.monotonic() if observed_at_monotonic is None else observed_at_monotonic
        entities = self.entities.update_tracks(tracked, camera_id=camera_id, now=now)
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
                    "last_verified_at": face.get("last_verified_at"),
                },
            ))
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
                metadata={"attendance_eligible": bool(face.get("attendance_eligible", False))},
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
            self._add(Observation(
                ObservationType.OBJECT_DETECTED,
                camera_id=camera_id,
                source_frame_id=object_frame_id,
                observed_at_monotonic=object_observed_at,
                observed_at_wallclock=object_wallclock,
                producer=str(detection.get("source_model") or "YOLO"),
                model_name=str(detection.get("source_model") or "YOLO"),
                confidence=detection.get("confidence"),
                confidence_type="detector_confidence",
                bbox=detection.get("bbox"),
                value=detection.get("class_name"),
                metadata={"category": detection.get("category"), "entity_association": "not_established"},
            ))

        self.entities.history.prune(now)
        self.global_history.prune(now)
        self.last_update_ms = (time.perf_counter() - started) * 1000.0
        return {
            "frame_id": source_frame_id,
            "camera_id": camera_id,
            "entities": self.entities.snapshot(now),
            "stats": self.stats(),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "frames": self.frames,
            "observations_created": self.observations_created,
            "global_observations": self.global_observations,
            "last_frame_id": self.last_frame_id,
            "last_update_ms": round(self.last_update_ms, 3),
            "expired_observations_rejected": self.entities.expired_observations_rejected,
            "entities": self.entities.stats(),
            "global_history": self.global_history.stats(),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "last_frame_id": self.last_frame_id,
            "last_update_ms": round(self.last_update_ms, 3),
            "stats": self.stats(),
            "entities": self.entities.snapshot(),
        }
