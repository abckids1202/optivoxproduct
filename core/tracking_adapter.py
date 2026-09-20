"""Tracking backend boundary used by person and vehicle entity pipelines."""

from __future__ import annotations

from typing import Any, Callable, Optional


class TrackingAdapter:
    """Keep tracker replacement separate from correlation and policy logic."""

    def __init__(self, backend: str, update_fn: Callable[[Any], Any], *, version: str = "local"):
        self.backend = str(backend or "unknown")
        self.version = str(version or "local")
        self._update_fn = update_fn
        self.update_calls = 0
        self.failures = 0
        self.last_error: Optional[str] = None
        self.frames = 0
        self.last_source_frame_id: Optional[int] = None
        self.last_active_track_count = 0
        self.membership_changes = 0
        self._previous_track_ids: set[int] = set()

    def update(self, detections: Any, *, source_frame_id: Optional[int] = None) -> Any:
        try:
            result = self._update_fn(detections)
            self.update_calls += 1
            self.frames += 1
            if source_frame_id is not None:
                self.last_source_frame_id = int(source_frame_id)
            if isinstance(result, dict):
                current_ids = {int(track_id) for track_id in result if str(track_id).lstrip("-").isdigit()}
                self.last_active_track_count = len(current_ids)
                if self.frames > 1:
                    self.membership_changes += len(current_ids.symmetric_difference(self._previous_track_ids))
                self._previous_track_ids = current_ids
            return result
        except Exception as exc:
            self.failures += 1
            self.last_error = str(exc)[:240]
            raise

    def snapshot(self, active_tracks: Optional[Any] = None) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "version": self.version,
            "status": "FAILED_RUNTIME" if self.failures >= 3 else "AVAILABLE",
            "update_calls": self.update_calls,
            "frames": self.frames,
            "last_source_frame_id": self.last_source_frame_id,
            "active_track_count": self.last_active_track_count,
            "membership_changes": self.membership_changes,
            "id_switches": "NOT_MEASURED",
            "measurement_status": "NOT_MEASURED",
            "failures": self.failures,
            "last_error": self.last_error,
            "active_tracks": len(active_tracks) if active_tracks is not None else None,
            "entity_generation_source": "CorrelationCore",
        }
