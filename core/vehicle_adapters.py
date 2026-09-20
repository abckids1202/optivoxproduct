"""Safe adapter contracts for optional vehicle perception components.

These contracts normalize a future plate detector/OCR implementation without
installing or trusting a model implicitly. They deliberately produce evidence
only; vehicle policy, incidents, access decisions, and attendance remain
outside this module.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .model_adapters import ModelSpec

_PLATE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 -]{1,15}$")


def normalize_plate_text(value: Any) -> Optional[str]:
    """Normalize plate text or return ``None`` without guessing characters."""
    normalized = " ".join(str(value or "").upper().split())
    return normalized if _PLATE_PATTERN.fullmatch(normalized) else None


@dataclass(frozen=True)
class PlateObservation:
    """A privacy-aware OCR result for one vehicle crop."""

    status: str
    track_id: Optional[int] = None
    text_local: Optional[str] = None
    text_hash: Optional[str] = None
    confidence: Optional[float] = None
    source_frame_id: Optional[int] = None
    reason: Optional[str] = None

    @classmethod
    def uncertain(cls, reason: str, *, track_id: Optional[int] = None,
                  source_frame_id: Optional[int] = None) -> "PlateObservation":
        return cls("PLATE_READ_UNCERTAIN", track_id=track_id,
                   source_frame_id=source_frame_id, reason=str(reason)[:160])

    def to_local_event(self, camera_id: str = "cam_0") -> dict[str, Any]:
        """Return an event payload with raw text explicitly marked local."""
        payload: dict[str, Any] = {
            "status": self.status,
            "track_id": self.track_id,
            "source_frame_id": self.source_frame_id,
            "entity_id": (
                f"{str(camera_id or 'cam_0')}:vehicle:{self.track_id}"
                if self.track_id is not None else None
            ),
            "privacy_class": "restricted_local",
            "entity_type": "vehicle",
        }
        if self.text_hash:
            payload["plate_hash"] = self.text_hash
        if self.text_local:
            payload["plate_text_local"] = self.text_local
        if self.confidence is not None:
            payload["ocr_confidence"] = self.confidence
        if self.reason:
            payload["reason"] = self.reason
        return payload


class PlateTemporalVoter:
    """Require repeated, consistent OCR readings before declaring a plate."""

    def __init__(self, min_hits: int = 3, history_size: int = 12,
                 min_confidence: float = 0.75):
        self.min_hits = max(2, int(min_hits))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self._history: deque[tuple[str, float, Optional[int]]] = deque(
            maxlen=max(self.min_hits, int(history_size)))

    def add(self, text: Any, confidence: Any,
            source_frame_id: Optional[int] = None) -> PlateObservation:
        normalized = normalize_plate_text(text)
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if normalized is None:
            return PlateObservation.uncertain("invalid_plate_text",
                                              source_frame_id=source_frame_id)
        if score < self.min_confidence:
            return PlateObservation.uncertain("low_ocr_confidence",
                                              source_frame_id=source_frame_id)
        self._history.append((normalized, score, source_frame_id))
        counts = Counter(item[0] for item in self._history)
        candidate, hits = counts.most_common(1)[0]
        if hits < self.min_hits:
            return PlateObservation.uncertain("awaiting_temporal_confirmation",
                                              source_frame_id=source_frame_id)
        readings = [item for item in self._history if item[0] == candidate]
        average = sum(item[1] for item in readings) / len(readings)
        return PlateObservation(
            "PLATE_READ", text_local=candidate,
            text_hash=hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            confidence=round(average, 4), source_frame_id=source_frame_id,
        )


class PlateOCRAdapter:
    """Fail-closed boundary for an optional licensed detector/OCR pipeline."""

    def __init__(self, spec: ModelSpec,
                 infer_fn: Optional[Callable[..., Any]] = None):
        if spec.task not in {"license_plate_reading", "plate_ocr"}:
            raise ValueError("PlateOCRAdapter requires a plate OCR task")
        self.spec = spec
        self.infer_fn = infer_fn
        self._voters: dict[Any, PlateTemporalVoter] = {}

    def capability_state(self) -> str:
        if not self.spec.enabled:
            return "DISABLED"
        if self.spec.checksum_verified is False:
            return "FAILED_INTEGRITY"
        if self.infer_fn is None:
            return "NOT_CONFIGURED"
        return self.spec.capability

    def infer(self, crop: Any, context: Optional[dict[str, Any]] = None) -> PlateObservation:
        state = self.capability_state()
        context = context or {}
        track_id = context.get("track_id")
        source_frame_id = context.get("source_frame_id")
        if state not in {"AVAILABLE", "EXPERIMENTAL"}:
            return PlateObservation.uncertain(
                state.lower(), track_id=track_id, source_frame_id=source_frame_id)
        try:
            result = self.infer_fn(crop, context)
        except Exception:
            return PlateObservation.uncertain(
                "ocr_runtime_failure", track_id=track_id,
                source_frame_id=source_frame_id)
        if not isinstance(result, dict):
            return PlateObservation.uncertain(
                "invalid_ocr_output", track_id=track_id,
                source_frame_id=source_frame_id)
        result_track = result.get("track_id", track_id)
        result_frame = result.get("source_frame_id", source_frame_id)
        voter_key = result_track if result_track is not None else "default"
        voter = self._voters.setdefault(voter_key, PlateTemporalVoter(min_hits=2))
        if len(self._voters) > 128:
            self._voters.pop(next(iter(self._voters)))
        observation = voter.add(
            result.get("text"), result.get("confidence", 0.0), result_frame)
        if observation.status == "PLATE_READ":
            return PlateObservation(
                observation.status, track_id=result_track,
                text_local=observation.text_local,
                text_hash=observation.text_hash,
                confidence=observation.confidence,
                source_frame_id=result_frame,
            )
        return PlateObservation.uncertain(
            observation.reason or "uncertain_ocr",
            track_id=result_track, source_frame_id=result_frame)
