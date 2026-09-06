"""Explicit per-track entity state for OptiVox correlation."""

from __future__ import annotations

import math
import time
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


@dataclass
class LivenessState:
    state: str = "NOT_EVALUATED"
    last_checked_monotonic: Optional[float] = None
    last_verified_monotonic: Optional[float] = None


@dataclass
class EntityState:
    entity_id: str
    entity_type: str
    camera_id: str
    track_id: int
    created_monotonic: float
    first_seen_monotonic: float
    last_seen_monotonic: float
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
        self.recent_observation_count += 1
        self.last_seen_monotonic = max(self.last_seen_monotonic, observation.observed_at_monotonic)
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
                self.identity.confirmation_hits = (
                    self.identity.confirmation_hits + 1 if same_candidate else 1
                )
                if (previous_state == IdentityDecisionState.CONFIRMED.value
                        and not revalidation_required):
                    self.identity.confirmation_hits = max(
                        self.identity.confirmation_hits,
                        self.identity_confirmation_observations,
                    )
                if self.identity.confirmation_hits >= self.identity_confirmation_observations:
                    self.identity.state = IdentityDecisionState.CONFIRMED.value
                    self.identity.confirmed_name = candidate_name
                else:
                    self.identity.state = IdentityDecisionState.CANDIDATE.value
            else:
                self.identity.state = incoming_state
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
            self.liveness.state = str(observation.value or "NOT_EVALUATED")
            self.liveness.last_checked_monotonic = observation.observed_at_monotonic
            if self.liveness.state == "REAL":
                self.liveness.last_verified_monotonic = observation.observed_at_monotonic
            elif self.liveness.state in {"SUSPECT", "SPOOF_SUSPECT"}:
                self.identity.state = IdentityDecisionState.SPOOF_SUSPECT.value
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
            elif self.identity.state == IdentityDecisionState.CONTRADICTED.value and identity_age > self.identity_expired_after_sec:
                self.identity.state = IdentityDecisionState.EXPIRED.value
        self.attendance_eligibility = bool(
            self.lifecycle_state in {EntityLifecycle.NEW, EntityLifecycle.ACTIVE}
            and self.face_visible
            and quality_current
            and self.quality_ok
            and self.identity.state == IdentityDecisionState.CONFIRMED.value
            and liveness_current
            and self.liveness.state == "REAL"
        )

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
            },
            "pose_state": self.pose_state,
            "current_zone": self.current_zone,
            "security_signals": list(self.security_signals),
            "attendance_eligibility": self.attendance_eligibility,
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
        self.track_switch_distance = max(50.0, float(track_switch_distance))
        self.entities: Dict[int, EntityState] = {}
        self.history = ObservationHistory(max_per_type=max_observations_per_type, max_entities=max_entities)
        self._next_entity_number = 1
        self._track_generations: Dict[int, int] = {}
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
        active_ids = {int(track_id) for track_id in tracked}
        for track_id, center in tracked.items():
            track_id = int(track_id)
            entity = self.entities.get(track_id)
            if entity is not None and entity.motion.current_center is not None:
                jump = math.hypot(
                    float(center[0]) - entity.motion.current_center[0],
                    float(center[1]) - entity.motion.current_center[1],
                )
                elapsed = max(0.0, current - entity.last_seen_monotonic)
                if jump > self.track_switch_distance and (elapsed > 0.5 or entity.missing_updates > 0):
                    entity.lifecycle_state = EntityLifecycle.CLOSED
                    self.history.clear_entity(track_id)
                    self.entities.pop(track_id, None)
                    entity = None
            entity = entity or self._new_entity(track_id, camera_id, current)
            entity.camera_id = camera_id
            entity.lifecycle_state = EntityLifecycle.ACTIVE
            entity.missing_updates = 0
            entity.last_seen_monotonic = current
            entity.update_center(center, current)
        for track_id, entity in list(self.entities.items()):
            if track_id in active_ids:
                continue
            entity.missing_updates += 1
            elapsed = max(0.0, current - entity.last_seen_monotonic)
            if elapsed >= self.close_after_sec:
                entity.lifecycle_state = EntityLifecycle.CLOSED
                self.history.clear_entity(track_id)
                self.entities.pop(track_id, None)
                self.closed_entities += 1
            elif entity.missing_updates >= self.occluded_after_updates:
                entity.lifecycle_state = EntityLifecycle.OCCLUDED
            else:
                entity.lifecycle_state = EntityLifecycle.STALE
        if len(self.entities) > self.max_entities:
            removable = sorted(
                self.entities.values(),
                key=lambda item: (item.lifecycle_state == EntityLifecycle.ACTIVE,
                                  item.last_seen_monotonic),
            )
            for entity in removable[:len(self.entities) - self.max_entities]:
                self.history.clear_entity(entity.track_id)
                self.entities.pop(entity.track_id, None)
                self.closed_entities += 1
        self.history.prune(current)
        return [self.entities[track_id] for track_id in sorted(active_ids) if track_id in self.entities]

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
