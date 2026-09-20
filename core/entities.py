"""Explicit per-track entity state for OptiVox correlation."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from .observations import Observation, ObservationHistory, ObservationType


class EntityLifecycle(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    OCCLUDED = "OCCLUDED"
    STALE = "STALE"
    LOST = "LOST"
    CLOSED = "CLOSED"


class IdentityDecisionState(str, Enum):
    """Canonical identity states shared by the runtime and platform layers."""

    UNRESOLVED = "UNRESOLVED"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    OCCLUDED = "OCCLUDED"
    EXPIRED = "EXPIRED"
    SPOOF_SUSPECT = "SPOOF_SUSPECT"


_IDENTITY_STATE_ALIASES = {
    "STALE": IdentityDecisionState.OCCLUDED.value,
    "REVALIDATING": IdentityDecisionState.CANDIDATE.value,
    "SPOOF": IdentityDecisionState.SPOOF_SUSPECT.value,
    "SUSPECT": IdentityDecisionState.SPOOF_SUSPECT.value,
    "UNKNOWN": IdentityDecisionState.UNRESOLVED.value,
}


def normalize_identity_state(value: object) -> str:
    """Normalize legacy runtime labels without weakening attendance rules."""
    state = str(value or IdentityDecisionState.UNRESOLVED.value).strip().upper()
    state = _IDENTITY_STATE_ALIASES.get(state, state)
    if state not in {item.value for item in IdentityDecisionState}:
        return IdentityDecisionState.UNRESOLVED.value
    return state


@dataclass
class MotionState:
    previous_center: Optional[Tuple[float, float]] = None
    current_center: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    last_updated_monotonic: float = 0.0


@dataclass
class IdentityState:
    state: str = IdentityDecisionState.UNRESOLVED.value
    candidate_name: Optional[str] = None
    confirmed_name: Optional[str] = None
    best_score: Optional[float] = None
    second_score: Optional[float] = None
    margin: Optional[float] = None
    last_verified_monotonic: Optional[float] = None
    last_observation_monotonic: Optional[float] = None
    contradiction_count: int = 0
    occluded_since_monotonic: Optional[float] = None
    confirmation_hits: int = 0
    # Cached display output is useful for continuity, but it is not fresh
    # biometric evidence and must never authorize a new attendance decision.
    current_evidence_fresh: bool = False
    last_cached_observation_monotonic: Optional[float] = None
    # Bounded voting history prevents one lucky frame from confirming a name.
    # The reducer still requires the current candidate to be the one with the
    # required number of votes in this window.
    candidate_window: deque = field(default_factory=lambda: deque(maxlen=5), repr=False)


@dataclass
class LivenessState:
    state: str = "NOT_EVALUATED"
    last_checked_monotonic: Optional[float] = None
    last_verified_monotonic: Optional[float] = None
    challenge_phase: Optional[str] = None
    failure_reason: Optional[str] = None
    challenge_attempts: int = 0


@dataclass
class EntityState:
    entity_id: str
    entity_type: str
    camera_id: str
    track_id: int
    created_monotonic: float
    first_seen_monotonic: float
    last_seen_monotonic: float
    last_source_frame_id: Optional[int] = None
    last_observed_wallclock: Optional[str] = None
    track_generation: int = 1
    lifecycle_state: EntityLifecycle = EntityLifecycle.NEW
    bbox: Optional[Tuple[float, float, float, float]] = None
    previous_bbox: Optional[Tuple[float, float, float, float]] = None
    track_confidence: Optional[float] = None
    motion: MotionState = field(default_factory=MotionState)
    current_zone: Optional[str] = None
    face_visible: bool = False
    last_face_observation_monotonic: Optional[float] = None
    face_quality: Optional[float] = None
    quality_ok: bool = False
    last_quality_observation_monotonic: Optional[float] = None
    yaw: Optional[float] = None
    identity: IdentityState = field(default_factory=IdentityState)
    liveness: LivenessState = field(default_factory=LivenessState)
    pose_state: Optional[str] = None
    last_pose_observation_monotonic: Optional[float] = None
    associated_objects: Tuple[str, ...] = ()
    security_signals: List[str] = field(default_factory=list)
    last_security_observation_monotonic: Optional[float] = None
    presence_started_monotonic: float = 0.0
    attendance_eligibility: bool = False
    missing_updates: int = 0
    recent_observation_count: int = 0
    identity_stale_after_sec: float = 3.0
    identity_expired_after_sec: float = 9.0
    identity_confirmation_observations: int = 1
    identity_confirmation_window: int = 5
    face_visible_after_sec: float = 1.0
    quality_valid_after_sec: float = 1.5
    liveness_valid_after_sec: float = 2.0

    @property
    def presence_duration_sec(self) -> float:
        return max(0.0, self.last_seen_monotonic - self.presence_started_monotonic)

    @property
    def identity_age_ms(self) -> Optional[float]:
        if self.identity.last_verified_monotonic is None:
            return None
        return max(0.0, (time.monotonic() - self.identity.last_verified_monotonic) * 1000.0)

    def update_center(self, center: Tuple[float, float], now: float) -> None:
        previous = self.motion.current_center
        self.motion.previous_center = previous
        self.motion.current_center = (float(center[0]), float(center[1]))
        if previous is not None and self.motion.last_updated_monotonic > 0:
            elapsed = max(1e-6, now - self.motion.last_updated_monotonic)
            self.motion.velocity = (
                (self.motion.current_center[0] - previous[0]) / elapsed,
                (self.motion.current_center[1] - previous[1]) / elapsed,
            )
            self.motion.speed = math.hypot(*self.motion.velocity)
        self.motion.last_updated_monotonic = now

    def attach(self, observation: Observation) -> None:
        previous_last_seen = self.last_seen_monotonic
        self.recent_observation_count += 1
        self.last_seen_monotonic = max(self.last_seen_monotonic, observation.observed_at_monotonic)
        if observation.source_frame_id is not None:
            if self.last_source_frame_id is None:
                self.last_source_frame_id = observation.source_frame_id
            else:
                self.last_source_frame_id = max(
                    int(self.last_source_frame_id), int(observation.source_frame_id))
        # Wall-clock provenance follows the same monotonic ordering as the
        # entity state. Delayed detector output must not make a live entity
        # appear to have been seen earlier than its already-recorded evidence.
        if (observation.observed_at_wallclock
                and observation.observed_at_monotonic >= previous_last_seen):
            self.last_observed_wallclock = observation.observed_at_wallclock
        if observation.observation_type == ObservationType.FACE_DETECTED:
            self.face_visible = True
            self.last_face_observation_monotonic = observation.observed_at_monotonic
            if observation.bbox:
                self.bbox = observation.bbox
        elif observation.observation_type == ObservationType.FACE_QUALITY:
            self.face_quality = observation.quality
            self.quality_ok = bool(observation.metadata.get("quality_ok", False))
            self.last_quality_observation_monotonic = observation.observed_at_monotonic
        elif observation.observation_type == ObservationType.FACE_HEAD_POSE:
            self.yaw = observation.metadata.get("yaw", observation.value)
        elif observation.observation_type == ObservationType.FACE_IDENTITY_RESULT:
            metadata = observation.metadata
            name = metadata.get("name") or observation.value
            candidate_name = str(name).strip() if name else None
            if candidate_name and (
                candidate_name.upper() in {"UNKNOWN", "SPOOF", "SPOOF_SUSPECT"}
                or candidate_name.upper().startswith("STRANGER_")
            ):
                candidate_name = None
            # A cache hit can be rendered as a name, but it is not new
            # recognition evidence. Preserve the last safe state for display
            # while making the current frame ineligible for attendance.
            if bool(metadata.get("cached_only")):
                self.identity.current_evidence_fresh = False
                self.identity.last_cached_observation_monotonic = observation.observed_at_monotonic
                return
            self.identity.current_evidence_fresh = True
            incoming_state = normalize_identity_state(metadata.get("identity_state"))
            previous_state = self.identity.state
            previous_candidate = self.identity.candidate_name
            if (
                self.identity.state == IdentityDecisionState.CONFIRMED.value
                and candidate_name
                and self.identity.confirmed_name
                and candidate_name.casefold() != self.identity.confirmed_name.casefold()
            ):
                incoming_state = IdentityDecisionState.CONTRADICTED.value
                self.identity.contradiction_count += 1
            elif (
                self.identity.state == IdentityDecisionState.CONFIRMED.value
                and not candidate_name
                and incoming_state in {
                    IdentityDecisionState.UNRESOLVED.value,
                    IdentityDecisionState.EXPIRED.value,
                }
            ):
                # A confirmed track receiving a genuine unknown/unresolved
                # result is contradictory evidence, not a harmless label
                # change. Attendance is blocked until the identity is rebuilt.
                incoming_state = IdentityDecisionState.CONTRADICTED.value
                self.identity.contradiction_count += 1
            self.identity.best_score = observation.confidence
            self.identity.second_score = metadata.get("second_score")
            self.identity.margin = metadata.get("margin")
            self.identity.last_observation_monotonic = observation.observed_at_monotonic
            self.identity.candidate_name = candidate_name

            if incoming_state in {
                IdentityDecisionState.CANDIDATE.value,
                IdentityDecisionState.CONFIRMED.value,
            } and candidate_name:
                same_candidate = bool(
                    previous_candidate
                    and previous_candidate.casefold() == candidate_name.casefold()
                )
                revalidation_required = previous_state in {
                    IdentityDecisionState.OCCLUDED.value,
                    IdentityDecisionState.EXPIRED.value,
                    IdentityDecisionState.CONTRADICTED.value,
                    IdentityDecisionState.SPOOF_SUSPECT.value,
                }
                if revalidation_required:
                    self.identity.confirmation_hits = 0
                    self.identity.candidate_window.clear()
                window_size = max(1, int(self.identity_confirmation_window))
                while len(self.identity.candidate_window) >= window_size:
                    self.identity.candidate_window.popleft()
                self.identity.candidate_window.append(candidate_name.casefold())
                self.identity.confirmation_hits = sum(
                    1 for item in self.identity.candidate_window
                    if item == candidate_name.casefold()
                )
                # A currently confirmed, continuously tracked identity is
                # retained through ordinary refresh evidence. Revalidation
                # after occlusion/contradiction still requires the full vote.
                continuously_confirmed = (
                    previous_state == IdentityDecisionState.CONFIRMED.value
                    and same_candidate
                    and not revalidation_required
                )
                if self.identity.confirmation_hits >= self.identity_confirmation_observations:
                    self.identity.state = IdentityDecisionState.CONFIRMED.value
                    self.identity.confirmed_name = candidate_name
                elif continuously_confirmed:
                    self.identity.state = IdentityDecisionState.CONFIRMED.value
                else:
                    self.identity.state = IdentityDecisionState.CANDIDATE.value
            else:
                self.identity.state = incoming_state
                self.identity.candidate_window.clear()
                if incoming_state in {
                    IdentityDecisionState.UNRESOLVED.value,
                    IdentityDecisionState.CONTRADICTED.value,
                    IdentityDecisionState.SPOOF_SUSPECT.value,
                }:
                    self.identity.confirmation_hits = 0

            if self.identity.state == IdentityDecisionState.CONFIRMED.value and candidate_name:
                self.identity.confirmed_name = candidate_name
                self.identity.last_verified_monotonic = observation.observed_at_monotonic
                self.identity.occluded_since_monotonic = None
            elif self.identity.state in {
                    IdentityDecisionState.CANDIDATE.value,
                    IdentityDecisionState.UNRESOLVED.value,
                } and self.identity.state != IdentityDecisionState.CONFIRMED.value:
                self.identity.occluded_since_monotonic = None
        elif observation.observation_type == ObservationType.SECURITY_SIGNAL:
            signal = str(
                observation.value
                or observation.metadata.get("event_type")
                or "SECURITY_SIGNAL"
            )
            if signal not in self.security_signals:
                self.security_signals.append(signal)
            self.security_signals = self.security_signals[-12:]
            self.last_security_observation_monotonic = observation.observed_at_monotonic
        elif observation.observation_type == ObservationType.LIVENESS_RESULT:
            liveness_state = str(observation.value or "NOT_EVALUATED").upper()
            self.liveness.state = (
                IdentityDecisionState.SPOOF_SUSPECT.value
                if liveness_state == "SUSPECT" else liveness_state
            )
            self.liveness.last_checked_monotonic = observation.observed_at_monotonic
            if self.liveness.state == "REAL":
                self.liveness.last_verified_monotonic = observation.observed_at_monotonic
            elif self.liveness.state in {"SUSPECT", "SPOOF_SUSPECT"}:
                self.identity.state = IdentityDecisionState.SPOOF_SUSPECT.value
                self.attendance_eligibility = False
            self.liveness.challenge_phase = observation.metadata.get("challenge_phase")
            self.liveness.failure_reason = observation.metadata.get("failure_reason")
            self.liveness.challenge_attempts = int(
                observation.metadata.get("challenge_attempts") or self.liveness.challenge_attempts or 0
            )
        elif observation.observation_type == ObservationType.POSE_STATE:
            self.pose_state = str(observation.value) if observation.value is not None else None
            self.last_pose_observation_monotonic = observation.observed_at_monotonic
        elif observation.observation_type == ObservationType.ZONE_MEMBERSHIP:
            self.current_zone = str(observation.value) if observation.value is not None else None
        elif observation.observation_type == ObservationType.TRACK_MOTION:
            velocity = observation.metadata.get("velocity")
            if velocity and len(velocity) >= 2:
                self.motion.velocity = (float(velocity[0]), float(velocity[1]))
                self.motion.speed = math.hypot(*self.motion.velocity)

    def refresh(self, now: Optional[float] = None) -> None:
        current = time.monotonic() if now is None else now
        self.face_visible = bool(
            self.last_face_observation_monotonic is not None
            and current - self.last_face_observation_monotonic <= self.face_visible_after_sec
        )
        quality_current = bool(
            self.last_quality_observation_monotonic is not None
            and current - self.last_quality_observation_monotonic <= self.quality_valid_after_sec
        )
        liveness_current = bool(
            self.liveness.last_checked_monotonic is not None
            and current - self.liveness.last_checked_monotonic <= self.liveness_valid_after_sec
        )
        if self.identity.last_observation_monotonic is not None:
            identity_age = current - self.identity.last_observation_monotonic
            if self.identity.state in {
                IdentityDecisionState.CONFIRMED.value,
                IdentityDecisionState.OCCLUDED.value,
            }:
                if identity_age > self.identity_stale_after_sec:
                    if self.identity.occluded_since_monotonic is None:
                        self.identity.occluded_since_monotonic = current
                    self.identity.confirmation_hits = 0
                    self.identity.state = (
                        IdentityDecisionState.EXPIRED.value
                        if identity_age > self.identity_expired_after_sec
                        else IdentityDecisionState.OCCLUDED.value
                    )
                    # The identity may remain useful as a display continuity
                    # label, but it is no longer current biometric evidence.
                    # Keep this distinction explicit for attendance, security
                    # attribution, and dashboard provenance.
                    self.identity.current_evidence_fresh = False
            elif self.identity.state in {
                    IdentityDecisionState.CANDIDATE.value,
                    IdentityDecisionState.CONTRADICTED.value,
            } and identity_age > self.identity_expired_after_sec:
                self.identity.state = IdentityDecisionState.EXPIRED.value
                self.identity.candidate_window.clear()
                self.identity.current_evidence_fresh = False
        self.attendance_eligibility = bool(
            self.lifecycle_state in {EntityLifecycle.NEW, EntityLifecycle.ACTIVE}
            and self.face_visible
            and quality_current
            and self.quality_ok
            and self.identity.state == IdentityDecisionState.CONFIRMED.value
            and self.identity.current_evidence_fresh
            and liveness_current
            and self.liveness.state == "REAL"
        )

    def attendance_gate(self, now: Optional[float] = None) -> Dict[str, object]:
        """Explain the local, non-roster portion of attendance eligibility."""
        current = time.monotonic() if now is None else now
        self.refresh(current)
        reasons: List[str] = []
        if self.lifecycle_state not in {EntityLifecycle.NEW, EntityLifecycle.ACTIVE}:
            reasons.append("entity_not_active")
        if not self.face_visible:
            reasons.append("face_not_visible")
        if not self.quality_ok or self.last_quality_observation_monotonic is None or current - self.last_quality_observation_monotonic > self.quality_valid_after_sec:
            reasons.append("WAITING_FOR_GOOD_FACE")
        if self.identity.state != IdentityDecisionState.CONFIRMED.value:
            reasons.append(f"identity_{self.identity.state.lower()}")
        if not self.identity.current_evidence_fresh:
            reasons.append("identity_evidence_cached_or_missing")
        if self.liveness.state != "REAL" or self.liveness.last_checked_monotonic is None or current - self.liveness.last_checked_monotonic > self.liveness_valid_after_sec:
            reasons.append("liveness_not_real")
        return {
            "eligible": not reasons,
            "reason": "eligible" if not reasons else ",".join(reasons),
            "identity_state": self.identity.state,
            "liveness_state": self.liveness.state,
            "quality_ok": self.quality_ok,
            "entity_state": self.lifecycle_state.value,
        }

    def summary(self, now: Optional[float] = None) -> Dict[str, object]:
        current = time.monotonic() if now is None else now
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "track_generation": getattr(self, "track_generation", 1),
            "lifecycle_state": self.lifecycle_state.value,
            "first_seen_monotonic": self.first_seen_monotonic,
            "last_seen_monotonic": self.last_seen_monotonic,
            "last_source_frame_id": self.last_source_frame_id,
            "last_observed_wallclock": self.last_observed_wallclock,
            "age_since_last_seen_ms": round(max(0.0, current - self.last_seen_monotonic) * 1000.0, 2),
            "bbox": self.bbox,
            "velocity": self.motion.velocity,
            "speed": round(self.motion.speed, 3),
            "identity": {
                "state": self.identity.state,
                "candidate_name": self.identity.candidate_name,
                "confirmed_name": self.identity.confirmed_name,
                "best_score": self.identity.best_score,
                "second_score": self.identity.second_score,
                "margin": self.identity.margin,
                "identity_age_ms": self.identity_age_ms,
                "contradiction_count": self.identity.contradiction_count,
                "occluded_since_monotonic": self.identity.occluded_since_monotonic,
                "confirmation_hits": self.identity.confirmation_hits,
                "current_evidence_fresh": self.identity.current_evidence_fresh,
                "last_cached_observation_monotonic": self.identity.last_cached_observation_monotonic,
                "candidate_window": list(self.identity.candidate_window),
            },
            "face": {
                "visible": self.face_visible,
                "quality": self.face_quality,
                "quality_ok": self.quality_ok,
                "yaw": self.yaw,
            },
            "liveness": {
                "state": self.liveness.state,
                "last_checked_monotonic": self.liveness.last_checked_monotonic,
                "last_verified_monotonic": self.liveness.last_verified_monotonic,
                "challenge_phase": self.liveness.challenge_phase,
                "failure_reason": self.liveness.failure_reason,
                "challenge_attempts": self.liveness.challenge_attempts,
            },
            "pose_state": self.pose_state,
            "current_zone": self.current_zone,
            "security_signals": list(self.security_signals),
            "attendance_eligibility": self.attendance_eligibility,
            "attendance_decision": self.attendance_gate(current),
            "recent_observation_count": self.recent_observation_count,
        }


class EntityStateStore:
    """Owns entity lifecycle and bounded observation history."""

    def __init__(self, max_entities: int = 128, max_observations_per_type: int = 12,
                 occluded_after_updates: int = 3, close_after_sec: float = 10.0,
                 identity_stale_after_sec: float = 3.0,
                 face_visible_after_sec: float = 1.0,
                 quality_valid_after_sec: float = 1.5,
                 liveness_valid_after_sec: float = 2.0,
                 identity_confirmation_observations: int = 1,
                 identity_confirmation_window: int = 5,
                 track_switch_distance: float = 250.0):
        self.max_entities = max(1, int(max_entities))
        self.occluded_after_updates = max(1, int(occluded_after_updates))
        self.close_after_sec = max(1.0, float(close_after_sec))
        self.identity_stale_after_sec = max(0.1, float(identity_stale_after_sec))
        self.face_visible_after_sec = max(0.2, float(face_visible_after_sec))
        self.quality_valid_after_sec = max(0.2, float(quality_valid_after_sec))
        self.liveness_valid_after_sec = max(0.2, float(liveness_valid_after_sec))
        self.identity_confirmation_observations = max(
            1, int(identity_confirmation_observations))
        self.identity_confirmation_window = max(
            self.identity_confirmation_observations, int(identity_confirmation_window))
        self.track_switch_distance = max(50.0, float(track_switch_distance))
        self.entities: Dict[int, EntityState] = {}
        self.history = ObservationHistory(max_per_type=max_observations_per_type, max_entities=max_entities)
        self._next_entity_number = 1
        self._track_generations: Dict[int, int] = {}
        self._last_transitions: List[Dict[str, object]] = []
        self.closed_entities = 0
        self.expired_observations_rejected = 0

    def _new_entity(self, track_id: int, camera_id: str, now: float) -> EntityState:
        entity = EntityState(
            entity_id=f"{camera_id}:entity:{self._next_entity_number}",
            entity_type="PERSON",
            camera_id=camera_id,
            track_id=int(track_id),
            created_monotonic=now,
            first_seen_monotonic=now,
            last_seen_monotonic=now,
            presence_started_monotonic=now,
            identity_stale_after_sec=self.identity_stale_after_sec,
            identity_expired_after_sec=max(self.identity_stale_after_sec, self.identity_stale_after_sec * 3.0),
            face_visible_after_sec=self.face_visible_after_sec,
            quality_valid_after_sec=self.quality_valid_after_sec,
            liveness_valid_after_sec=self.liveness_valid_after_sec,
            identity_confirmation_observations=self.identity_confirmation_observations,
            identity_confirmation_window=self.identity_confirmation_window,
        )
        generation = self._track_generations.get(int(track_id), 0) + 1
        self._track_generations[int(track_id)] = generation
        entity.track_generation = generation
        self._next_entity_number += 1
        self.entities[int(track_id)] = entity
        return entity

    def update_tracks(
        self,
        tracked: Dict[int, Tuple[int, int]],
        camera_id: str = "cam_0",
        now: Optional[float] = None,
    ) -> List[EntityState]:
        current = time.monotonic() if now is None else now
        self._last_transitions = []
        active_ids = {int(track_id) for track_id in tracked}
        for track_id, center in tracked.items():
            track_id = int(track_id)
            entity = self.entities.get(track_id)
            if entity is not None and str(entity.camera_id) != str(camera_id):
                # Numeric tracker IDs are local to a camera. Never mutate an
                # existing entity into another camera context, because that
                # would carry identity, liveness, and attendance evidence
                # across independent video streams.
                self._last_transitions.append({
                    "type": "closed",
                    "reason": "camera_changed",
                    "entity_id": entity.entity_id,
                    "track_id": entity.track_id,
                    "track_generation": entity.track_generation,
                    "camera_id": entity.camera_id,
                    "last_seen_monotonic": entity.last_seen_monotonic,
                })
                entity.lifecycle_state = EntityLifecycle.CLOSED
                self.history.clear_entity(track_id)
                self.entities.pop(track_id, None)
                entity = None
            if entity is not None and entity.motion.current_center is not None:
                jump = math.hypot(
                    float(center[0]) - entity.motion.current_center[0],
                    float(center[1]) - entity.motion.current_center[1],
                )
                elapsed = max(0.0, current - entity.last_seen_monotonic)
                if jump > self.track_switch_distance and (elapsed > 0.5 or entity.missing_updates > 0):
                    self._last_transitions.append({
                        "type": "closed",
                        "reason": "tracker_switch",
                        "entity_id": entity.entity_id,
                        "track_id": entity.track_id,
                        "track_generation": entity.track_generation,
                        "camera_id": entity.camera_id,
                        "last_seen_monotonic": entity.last_seen_monotonic,
                    })
                    entity.lifecycle_state = EntityLifecycle.CLOSED
                    self.history.clear_entity(track_id)
                    self.entities.pop(track_id, None)
                    entity = None
            was_new = entity is None
            entity = entity or self._new_entity(track_id, camera_id, current)
            if was_new:
                self._last_transitions.append({
                    "type": "opened",
                    "reason": "first_observation",
                    "entity_id": entity.entity_id,
                    "track_id": entity.track_id,
                    "track_generation": entity.track_generation,
                    "camera_id": entity.camera_id,
                    "started_monotonic": entity.first_seen_monotonic,
                })
            elif entity.lifecycle_state in {EntityLifecycle.STALE, EntityLifecycle.OCCLUDED}:
                self._last_transitions.append({
                    "type": "revalidated",
                    "reason": "track_visible_again",
                    "entity_id": entity.entity_id,
                    "track_id": entity.track_id,
                    "track_generation": entity.track_generation,
                    "camera_id": entity.camera_id,
                })
            entity.camera_id = camera_id
            entity.lifecycle_state = EntityLifecycle.ACTIVE
            entity.missing_updates = 0
            # A delayed frame may arrive after a newer frame has already
            # reduced state. Preserve temporal ordering and avoid deriving a
            # large backwards-time velocity from stale input.
            effective_now = max(current, entity.last_seen_monotonic)
            entity.last_seen_monotonic = effective_now
            entity.update_center(center, effective_now)
        for track_id, entity in list(self.entities.items()):
            if track_id in active_ids:
                continue
            entity.missing_updates += 1
            elapsed = max(0.0, current - entity.last_seen_monotonic)
            if elapsed >= self.close_after_sec:
                self._last_transitions.append({
                    "type": "closed",
                    "reason": "timeout",
                    "entity_id": entity.entity_id,
                    "track_id": entity.track_id,
                    "track_generation": entity.track_generation,
                    "camera_id": entity.camera_id,
                    "last_seen_monotonic": entity.last_seen_monotonic,
                })
                entity.lifecycle_state = EntityLifecycle.CLOSED
                self.history.clear_entity(track_id)
                self.entities.pop(track_id, None)
                self.closed_entities += 1
            elif entity.missing_updates >= self.occluded_after_updates:
                if entity.lifecycle_state != EntityLifecycle.OCCLUDED:
                    self._last_transitions.append({
                        "type": "occluded",
                        "reason": "temporary_track_loss",
                        "entity_id": entity.entity_id,
                        "track_id": entity.track_id,
                        "track_generation": entity.track_generation,
                        "camera_id": entity.camera_id,
                    })
                entity.lifecycle_state = EntityLifecycle.OCCLUDED
            else:
                # Preserve the short-gap lifecycle label for compatibility;
                # both STALE and OCCLUDED are ineligible for attendance.
                if entity.lifecycle_state == EntityLifecycle.ACTIVE:
                    self._last_transitions.append({
                        "type": "occluded",
                        "reason": "short_detection_gap",
                        "entity_id": entity.entity_id,
                        "track_id": entity.track_id,
                        "track_generation": entity.track_generation,
                        "camera_id": entity.camera_id,
                    })
                entity.lifecycle_state = EntityLifecycle.STALE
        if len(self.entities) > self.max_entities:
            removable = sorted(
                self.entities.values(),
                key=lambda item: (item.lifecycle_state == EntityLifecycle.ACTIVE,
                                  item.last_seen_monotonic),
            )
            for entity in removable[:len(self.entities) - self.max_entities]:
                self._last_transitions.append({
                    "type": "closed",
                    "reason": "entity_capacity",
                    "entity_id": entity.entity_id,
                    "track_id": entity.track_id,
                    "track_generation": entity.track_generation,
                    "camera_id": entity.camera_id,
                    "last_seen_monotonic": entity.last_seen_monotonic,
                })
                self.history.clear_entity(entity.track_id)
                self.entities.pop(entity.track_id, None)
                self.closed_entities += 1
        self.history.prune(current)
        return [self.entities[track_id] for track_id in sorted(active_ids) if track_id in self.entities]

    def consume_transitions(self) -> List[Dict[str, object]]:
        """Return and clear lifecycle transitions from the latest update.

        The persistence bridge uses these transitions to close durable
        sessions immediately. Keeping this queue bounded to one update avoids
        replaying old lifecycle changes when a worker is delayed.
        """
        transitions = list(self._last_transitions)
        self._last_transitions = []
        return transitions

    def attach(self, observation: Observation) -> Optional[EntityState]:
        if observation.entity_track_id is None:
            return None
        if observation.is_expired():
            self.expired_observations_rejected += 1
            return None
        entity = self.entities.get(int(observation.entity_track_id))
        if entity is None or entity.lifecycle_state == EntityLifecycle.CLOSED:
            return None
        self.history.add(observation)
        entity.attach(observation)
        return entity

    def get(self, track_id: int) -> Optional[EntityState]:
        return self.entities.get(int(track_id))

    def active(self) -> List[EntityState]:
        return [entity for entity in self.entities.values() if entity.lifecycle_state in {
            EntityLifecycle.NEW, EntityLifecycle.ACTIVE, EntityLifecycle.OCCLUDED, EntityLifecycle.STALE
        }]

    def enforce_identity_exclusivity(self) -> List[Dict[str, object]]:
        """Prevent one roster identity from authorizing two live entities.

        Nearest-centroid tracking can briefly duplicate a track or associate
        two faces with the same roster match. Preserving both confirmations
        would allow an identity error to reach attendance. The strongest
        current entity keeps the confirmation; competing entities are forced
        through fresh temporal validation. This is intentionally conservative:
        a false rejection is safer than two simultaneous attendance records.
        """
        candidates: Dict[str, List[EntityState]] = {}
        for entity in self.entities.values():
            if entity.lifecycle_state not in {EntityLifecycle.NEW, EntityLifecycle.ACTIVE}:
                continue
            identity = entity.identity
            if identity.state != IdentityDecisionState.CONFIRMED.value:
                continue
            if not identity.confirmed_name or not identity.current_evidence_fresh:
                continue
            key = identity.confirmed_name.strip().casefold()
            if key:
                candidates.setdefault(key, []).append(entity)

        conflicts: List[Dict[str, object]] = []
        for normalized_name, group in candidates.items():
            if len(group) < 2:
                continue

            def rank(entity: EntityState):
                identity = entity.identity
                return (
                    bool(entity.face_visible),
                    bool(entity.quality_ok),
                    entity.liveness.state == "REAL",
                    float(identity.best_score) if identity.best_score is not None else -1.0,
                    -int(entity.track_id),
                )

            winner = max(group, key=rank)
            for loser in sorted(group, key=lambda item: item.track_id):
                if loser is winner:
                    continue
                identity = loser.identity
                identity.state = IdentityDecisionState.CONTRADICTED.value
                identity.contradiction_count += 1
                identity.confirmation_hits = 0
                identity.candidate_window.clear()
                identity.current_evidence_fresh = False
                loser.attendance_eligibility = False
                conflicts.append({
                    "identity": winner.identity.confirmed_name,
                    "normalized_identity": normalized_name,
                    "winner_entity_id": winner.entity_id,
                    "winner_track_id": winner.track_id,
                    "winner_track_generation": winner.track_generation,
                    "loser_entity_id": loser.entity_id,
                    "loser_track_id": loser.track_id,
                    "loser_track_generation": loser.track_generation,
                    "reason": "simultaneous_identity_collision",
                })
        return conflicts

    def prune(self, now: Optional[float] = None) -> None:
        current = time.monotonic() if now is None else now
        self.update_tracks({}, now=current)

    def snapshot(self, now: Optional[float] = None) -> List[Dict[str, object]]:
        current = time.monotonic() if now is None else now
        for entity in self.active():
            latest_face = self.history.latest(entity.track_id, ObservationType.FACE_DETECTED, current)
            latest_quality = self.history.latest(entity.track_id, ObservationType.FACE_QUALITY, current)
            latest_identity = self.history.latest(entity.track_id, ObservationType.FACE_IDENTITY_RESULT, current)
            latest_liveness = self.history.latest(entity.track_id, ObservationType.LIVENESS_RESULT, current)
            latest_pose = self.history.latest(entity.track_id, ObservationType.POSE_STATE, current)
            entity.face_visible = latest_face is not None
            if latest_quality is None:
                entity.face_quality = None
            if latest_identity is None and entity.identity.state == IdentityDecisionState.CONFIRMED.value:
                entity.identity.state = IdentityDecisionState.OCCLUDED.value
                entity.attendance_eligibility = False
            if latest_liveness is None:
                entity.liveness.state = "NOT_EVALUATED"
            if latest_pose is None:
                entity.pose_state = None
            entity.refresh(current)
        return [entity.summary(current) for entity in sorted(self.active(), key=lambda item: item.track_id)]

    def stats(self) -> Dict[str, int]:
        return {
            "active_entities": len(self.entities),
            "closed_entities": self.closed_entities,
            "expired_observations_rejected": self.expired_observations_rejected,
            **self.history.stats(),
        }
