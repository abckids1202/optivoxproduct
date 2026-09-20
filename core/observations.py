"""Fresh, provenance-aware observations for the real-time vision pipeline.

Observations stay in memory. Durable event and attendance records remain the
responsibility of the existing operational/database layer.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


class ObservationType(str, Enum):
    PERSON_DETECTED = "PERSON_DETECTED"
    FACE_DETECTED = "FACE_DETECTED"
    FACE_QUALITY = "FACE_QUALITY"
    FACE_HEAD_POSE = "FACE_HEAD_POSE"
    FACE_IDENTITY_RESULT = "FACE_IDENTITY_RESULT"
    LIVENESS_RESULT = "LIVENESS_RESULT"
    POSE_STATE = "POSE_STATE"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    SECURITY_SIGNAL = "SECURITY_SIGNAL"
    ZONE_MEMBERSHIP = "ZONE_MEMBERSHIP"
    TRACK_MOTION = "TRACK_MOTION"


# These are deliberately conservative defaults. They define a safety boundary,
# not a recognition policy. The runtime can override them per observation type.
DEFAULT_TTL_MS: Dict[ObservationType, int] = {
    ObservationType.PERSON_DETECTED: 350,
    ObservationType.FACE_DETECTED: 350,
    ObservationType.FACE_QUALITY: 750,
    ObservationType.FACE_HEAD_POSE: 500,
    ObservationType.FACE_IDENTITY_RESULT: 3000,
    ObservationType.LIVENESS_RESULT: 1500,
    ObservationType.POSE_STATE: 350,
    ObservationType.OBJECT_DETECTED: 750,
    ObservationType.SECURITY_SIGNAL: 1500,
    ObservationType.ZONE_MEMBERSHIP: 500,
    ObservationType.TRACK_MOTION: 500,
}

_OBSERVATION_COUNTER = itertools.count(1)


def monotonic_now() -> float:
    return time.monotonic()


def wallclock_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _coerce_type(value: ObservationType | str) -> ObservationType:
    if isinstance(value, ObservationType):
        return value
    return ObservationType(str(value))


@dataclass
class Observation:
    """One bounded, time-aware statement produced by one subsystem."""

    observation_type: ObservationType | str
    camera_id: str = "cam_0"
    source_frame_id: Optional[int] = None
    observed_at_monotonic: float = field(default_factory=monotonic_now)
    observed_at_wallclock: str = field(default_factory=wallclock_now)
    expires_at_monotonic: Optional[float] = None
    producer: str = "unknown"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    entity_track_id: Optional[int] = None
    confidence: Optional[float] = None
    confidence_type: Optional[str] = None
    quality: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    geometry: Any = None
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_ms: Optional[int] = None
    observation_id: str = field(default_factory=lambda: f"obs-{time.monotonic_ns()}-{next(_OBSERVATION_COUNTER)}")

    def __post_init__(self) -> None:
        self.observation_type = _coerce_type(self.observation_type)
        if self.expires_at_monotonic is None:
            ttl = self.ttl_ms if self.ttl_ms is not None else DEFAULT_TTL_MS.get(self.observation_type, 500)
            self.expires_at_monotonic = self.observed_at_monotonic + max(1, int(ttl)) / 1000.0
        if self.bbox is not None:
            self.bbox = tuple(float(value) for value in self.bbox[:4])

    def age_ms(self, now: Optional[float] = None) -> float:
        current = monotonic_now() if now is None else now
        return max(0.0, (current - self.observed_at_monotonic) * 1000.0)

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = monotonic_now() if now is None else now
        return current >= float(self.expires_at_monotonic or current)

    def to_summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "type": self.observation_type.value,
            "camera_id": self.camera_id,
            "source_frame_id": self.source_frame_id,
            "age_ms": round(self.age_ms(now), 2),
            "expired": self.is_expired(now),
            "producer": self.producer,
            "model_name": self.model_name,
            "entity_track_id": self.entity_track_id,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "quality": self.quality,
        }


class ObservationHistory:
    """Bounded per-entity/per-type observation history."""

    def __init__(self, max_per_type: int = 12, max_entities: int = 128):
        self.max_per_type = max(1, int(max_per_type))
        self.max_entities = max(1, int(max_entities))
        self._buckets: Dict[Tuple[Optional[int], ObservationType], Deque[Observation]] = defaultdict(
            lambda: deque(maxlen=self.max_per_type)
        )
        self._last_touched: OrderedDict[Tuple[Optional[int], ObservationType], None] = OrderedDict()

    def add(self, observation: Observation) -> Observation:
        key = (observation.entity_track_id, observation.observation_type)
        self._buckets[key].append(observation)
        self._last_touched.pop(key, None)
        self._last_touched[key] = None
        self.prune()
        return observation

    def recent(
        self,
        entity_track_id: Optional[int],
        observation_type: ObservationType | str,
        now: Optional[float] = None,
        include_expired: bool = False,
    ) -> List[Observation]:
        key = (entity_track_id, _coerce_type(observation_type))
        values = list(self._buckets.get(key, ()))
        if include_expired:
            return values
        return [item for item in values if not item.is_expired(now)]

    def latest(
        self,
        entity_track_id: Optional[int],
        observation_type: ObservationType | str,
        now: Optional[float] = None,
    ) -> Optional[Observation]:
        values = self.recent(entity_track_id, observation_type, now=now)
        return values[-1] if values else None

    def prune(self, now: Optional[float] = None) -> None:
        current = monotonic_now() if now is None else now
        for key in list(self._buckets):
            bucket = self._buckets[key]
            while bucket and bucket[0].is_expired(current):
                bucket.popleft()
            if not bucket:
                self._buckets.pop(key, None)
                self._last_touched.pop(key, None)
        # ``max_entities`` limits tracked entity IDs, not the number of
        # observation-type buckets. Each entity legitimately owns several
        # buckets (face, quality, identity, liveness, motion, and so on);
        # evicting individual buckets could remove identity evidence while
        # retaining the corresponding face evidence and create false state.
        entity_ids = {
            key[0] for key in self._buckets
            if key[0] is not None
        }
        while len(entity_ids) > self.max_entities:
            oldest_entity = next(
                (key[0] for key in self._last_touched if key[0] is not None),
                None,
            )
            if oldest_entity is None:
                break
            for key in list(self._buckets):
                if key[0] == oldest_entity:
                    self._buckets.pop(key, None)
                    self._last_touched.pop(key, None)
            entity_ids.discard(oldest_entity)

    def clear_entity(self, entity_track_id: Optional[int]) -> None:
        for key in list(self._buckets):
            if key[0] == entity_track_id:
                self._buckets.pop(key, None)
                self._last_touched.pop(key, None)

    def stats(self) -> Dict[str, int]:
        return {
            "buckets": len(self._buckets),
            "observations": sum(len(bucket) for bucket in self._buckets.values()),
            "max_per_type": self.max_per_type,
            "max_entities": self.max_entities,
        }
