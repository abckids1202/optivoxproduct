"""Calibrated vehicle tracking and temporal plate-evidence boundary.

This module intentionally does not pretend to perform plate OCR. It accepts
vehicle detections from the existing detector, computes physical speed only
when a valid image-to-ground homography is configured, and provides a safe
adapter for a future licensed plate detector/OCR model.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .vehicle_adapters import PlateObservation, normalize_plate_text


VEHICLE_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "motorbike", "van", "vehicle",
})


@dataclass(frozen=True)
class CameraCalibration:
    homography: Tuple[Tuple[float, float, float], ...]
    image_width: int
    image_height: int
    version: str = "1"

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> Optional["CameraCalibration"]:
        if not isinstance(value, dict):
            return None
        matrix = value.get("homography")
        try:
            rows = tuple(tuple(float(item) for item in row) for row in matrix)
            width = int(value.get("image_width"))
            height = int(value.get("image_height"))
        except (TypeError, ValueError):
            return None
        if len(rows) != 3 or any(len(row) != 3 for row in rows) or width <= 0 or height <= 0:
            return None
        array = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(array).all() or abs(float(np.linalg.det(array))) < 1e-9:
            return None
        return cls(rows, width, height, str(value.get("version") or "1"))

    def project(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        vector = np.asarray([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
        projected = np.asarray(self.homography, dtype=np.float64) @ vector
        if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-9:
            return None
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])

    def to_dict(self) -> dict:
        return {
            "homography": [list(row) for row in self.homography],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "version": self.version,
        }


@dataclass
class VehicleTrack:
    track_id: int
    class_name: str
    center: Tuple[float, float]
    last_seen: float
    first_seen: float
    source_frame_id: Optional[int] = None
    ground_point: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed_mps: Optional[float] = None
    speed_history: deque = field(default_factory=lambda: deque(maxlen=8))
    plates: deque = field(default_factory=lambda: deque(maxlen=12))
    speed_alerted_at: Optional[float] = None
    speed_reported_at: Optional[float] = None


class VehicleIntelligence:
    """Track vehicles and emit conservative, attributable vehicle events."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("ENABLED", True))
        self.max_match_distance = max(10.0, float(self.cfg.get("MAX_MATCH_DISTANCE_PX", 140.0)))
        self.max_prediction_sec = max(0.0, float(self.cfg.get("MAX_PREDICTION_SEC", 0.75)))
        self.track_ttl_sec = max(0.25, float(self.cfg.get("TRACK_TTL_SEC", 2.0)))
        self.speed_limit_kmh = self.cfg.get("SPEED_LIMIT_KMH")
        self.speed_confirm_sec = max(0.1, float(self.cfg.get("SPEED_CONFIRM_SECONDS", 0.8)))
        self.speed_cooldown_sec = max(1.0, float(self.cfg.get("SPEED_ALERT_COOLDOWN_SECONDS", 30.0)))
        self.speed_report_interval_sec = max(
            0.2, float(self.cfg.get("SPEED_REPORT_INTERVAL_SECONDS", 1.0)))
        self.plate_min_confidence = max(0.0, min(1.0, float(self.cfg.get("PLATE_MIN_CONFIDENCE", 0.75))))
        self.plate_min_hits = max(2, int(self.cfg.get("PLATE_MIN_HITS", 3)))
        self.calibration = CameraCalibration.from_dict(self.cfg.get("CALIBRATION"))
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self._speed_started: Dict[int, float] = {}

    def set_calibration(self, value: Optional[dict]) -> bool:
        calibration = CameraCalibration.from_dict(value)
        if value is not None and calibration is None:
            return False
        self.calibration = calibration
        return True

    def capability_state(self) -> Dict[str, str]:
        if not self.enabled:
            return {"vehicle_tracking": "DISABLED", "vehicle_speed": "DISABLED", "plate_reading": "DISABLED"}
        return {
            "vehicle_tracking": "AVAILABLE",
            "vehicle_speed": "AVAILABLE" if self.calibration else "NOT_CONFIGURED",
            "plate_reading": "NOT_CONFIGURED",
            "plate_temporal_voting": "AVAILABLE",
        }

    @staticmethod
    def _anchor(detection: dict) -> Tuple[float, float]:
        box = detection.get("bbox") or (0, 0, 0, 0)
        return ((float(box[0]) + float(box[2])) * 0.5, float(box[3]))

    @staticmethod
    def _is_vehicle(detection: dict) -> bool:
        name = str(detection.get("class_name") or "").casefold()
        return name in VEHICLE_CLASSES or str(detection.get("category") or "").casefold() == "vehicle"

    def _calibration_point(self, point: Tuple[float, float], frame_size=None) -> Tuple[float, float]:
        """Map detector coordinates into the calibration image space."""
        if not self.calibration or not frame_size:
            return point
        try:
            source_width, source_height = float(frame_size[0]), float(frame_size[1])
            if source_width <= 0 or source_height <= 0:
                return point
            return (
                point[0] * self.calibration.image_width / source_width,
                point[1] * self.calibration.image_height / source_height,
            )
        except (TypeError, ValueError, IndexError, OverflowError):
            return point

    def _assign(self, detections: List[dict], now: float) -> Dict[int, dict]:
        available = set(self.tracks)
        assignment: Dict[int, dict] = {}
        for detection in sorted(detections, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
            center = self._anchor(detection)
            detection_class = str(detection.get("class_name") or "vehicle").casefold()
            choices = []
            for item in available:
                track = self.tracks[item]
                track_class = str(track.class_name or "vehicle").casefold()
                if (detection_class != track_class
                        and detection_class not in {"vehicle", ""}
                        and track_class not in {"vehicle", ""}):
                    continue
                elapsed = min(self.max_prediction_sec, max(0.0, float(now) - track.last_seen))
                predicted = (
                    track.center[0] + track.velocity[0] * elapsed,
                    track.center[1] + track.velocity[1] * elapsed,
                )
                choices.append((
                    math.hypot(center[0] - predicted[0], center[1] - predicted[1]),
                    item,
                ))
            if choices:
                distance, track_id = min(choices)
            else:
                distance, track_id = float("inf"), None
            if track_id is None or distance > self.max_match_distance:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = VehicleTrack(
                    track_id, str(detection.get("class_name") or "vehicle"), center, now, now)
            available.discard(track_id)
            assignment[track_id] = detection
        return assignment

    def update(self, detections: Iterable[dict], now: float, source_frame_id: Optional[int] = None,
               frame_size=None, camera_id: str = "cam_0") -> List[tuple]:
        if not self.enabled:
            return []
        current = float(now)
        usable = [item for item in detections if isinstance(item, dict) and self._is_vehicle(item)]
        assignment = self._assign(usable, current)
        events: List[tuple] = []
        for track_id, detection in assignment.items():
            track = self.tracks[track_id]
            vehicle_entity_id = f"{str(camera_id or 'cam_0')}:vehicle:{track_id}"
            previous_point = track.ground_point
            previous_seen = track.last_seen
            previous_center = track.center
            center = self._anchor(detection)
            ground = self.calibration.project(self._calibration_point(center, frame_size)) if self.calibration else None
            track.center = center
            track.last_seen = current
            track.source_frame_id = source_frame_id
            track.ground_point = ground
            elapsed = current - previous_seen
            if elapsed > 0:
                measured_velocity = (
                    (center[0] - previous_center[0]) / elapsed,
                    (center[1] - previous_center[1]) / elapsed,
                )
                if all(math.isfinite(value) for value in measured_velocity):
                    track.velocity = (
                        track.velocity[0] * 0.35 + measured_velocity[0] * 0.65,
                        track.velocity[1] * 0.35 + measured_velocity[1] * 0.65,
                    )
            if previous_point is not None and ground is not None and current > previous_seen:
                speed = math.hypot(ground[0] - previous_point[0], ground[1] - previous_point[1]) / (current - previous_seen)
                if math.isfinite(speed):
                    track.speed_mps = speed
                    track.speed_history.append(speed)
            if (self.calibration and len(track.speed_history) >= 2
                    and (track.speed_reported_at is None
                         or current - track.speed_reported_at >= self.speed_report_interval_sec)):
                average_mps = sum(track.speed_history) / len(track.speed_history)
                if math.isfinite(average_mps):
                    track.speed_reported_at = current
                    events.append((
                        "SPEED_ESTIMATE", f"VEHICLE_{track_id}",
                        float(detection.get("confidence") or 0.0),
                        f"Vehicle track {track_id} has a calibrated speed estimate.",
                        {
                            "track_id": track_id,
                            "speed_mps": round(average_mps, 3),
                            "speed_kmh": round(average_mps * 3.6, 2),
                            "calibrated": True,
                            "calibration_version": self.calibration.version,
                            "source_frame_id": source_frame_id,
                            "entity_id": vehicle_entity_id,
                            "entity_type": "vehicle",
                        },
                    ))
            if current == track.first_seen:
                events.append((
                    "VEHICLE_ENTERED", f"VEHICLE_{track_id}",
                    float(detection.get("confidence") or 0.0),
                    f"Vehicle track {track_id} entered the camera view.",
                    {"track_id": track_id, "source_frame_id": source_frame_id,
                     "class_name": track.class_name, "calibrated": bool(self.calibration),
                     "entity_id": vehicle_entity_id, "camera_id": str(camera_id or "cam_0"),
                     "entity_type": "vehicle"},
                ))
            limit = None
            try:
                limit = float(self.speed_limit_kmh) if self.speed_limit_kmh is not None else None
            except (TypeError, ValueError):
                limit = None
            if limit is not None and track.speed_mps is not None and self.calibration:
                average_kmh = sum(track.speed_history) / max(1, len(track.speed_history)) * 3.6
                if average_kmh >= limit:
                    self._speed_started.setdefault(track_id, current)
                    if (current - self._speed_started[track_id] >= self.speed_confirm_sec
                            and (track.speed_alerted_at is None or current - track.speed_alerted_at >= self.speed_cooldown_sec)):
                        track.speed_alerted_at = current
                        events.append((
                            "SPEED_THRESHOLD_EXCEEDED", f"VEHICLE_{track_id}",
                            min(1.0, average_kmh / max(limit, 1.0)),
                            f"Vehicle track {track_id} exceeded the configured speed threshold.",
                            {"track_id": track_id, "speed_kmh": round(average_kmh, 2),
                            "speed_limit_kmh": limit, "calibrated": True,
                             "source_frame_id": source_frame_id, "entity_id": vehicle_entity_id,
                             "entity_type": "vehicle"},
                        ))
                else:
                    self._speed_started.pop(track_id, None)
        for track_id in list(self.tracks):
            if current - self.tracks[track_id].last_seen > self.track_ttl_sec:
                track = self.tracks.pop(track_id, None)
                if track is not None:
                    # A timeout is the only defensible exit signal available
                    # without a second camera or an explicit exit line. Keep
                    # it attributable and make the timeout visible to the
                    # incident/persistence layers.
                    events.append((
                        "VEHICLE_EXITED", f"VEHICLE_{track_id}", 1.0,
                        f"Vehicle track {track_id} left the camera view.",
                        {
                            "track_id": track_id,
                            "source_frame_id": track.source_frame_id,
                            "last_seen_at": track.last_seen,
                            "class_name": track.class_name,
                            "calibrated": bool(self.calibration),
                            "exit_reason": "track_timeout",
                            "entity_id": f"{str(camera_id or 'cam_0')}:vehicle:{track_id}",
                            "camera_id": str(camera_id or "cam_0"),
                            "entity_type": "vehicle",
                        },
                    ))
                self._speed_started.pop(track_id, None)
        return events

    def ingest_plate(self, track_id: int, text: str, confidence: float, now: float,
                     source_frame_id: Optional[int] = None,
                     camera_id: str = "cam_0") -> Optional[tuple]:
        """Accept a future OCR result only after syntax, confidence, and voting."""
        track = self.tracks.get(int(track_id))
        candidate = normalize_plate_text(text)
        if track is None or candidate is None or float(confidence) < self.plate_min_confidence:
            return None
        track.plates.append((candidate, float(confidence), float(now), source_frame_id))
        counts = Counter(item[0] for item in track.plates)
        plate, hits = counts.most_common(1)[0]
        if hits < self.plate_min_hits:
            return None
        weighted_confidence = sum(item[1] for item in track.plates if item[0] == plate) / hits
        observation = PlateObservation(
            "PLATE_READ", track_id=int(track_id), text_local=plate,
            text_hash=hashlib.sha256(plate.encode("utf-8")).hexdigest(),
            source_frame_id=source_frame_id,
        )
        metadata = observation.to_local_event(camera_id)
        return (
            observation.status, f"VEHICLE_{track_id}", weighted_confidence,
            f"Vehicle track {track_id} produced a stable plate reading.",
            {**metadata, "track_id": int(track_id)},
        )

    def plate_unreadable(self, track_id: int, reason: str = "insufficient_quality",
                         source_frame_id: Optional[int] = None,
                         camera_id: str = "cam_0") -> Optional[tuple]:
        """Return an explicit uncertainty event without inventing plate text.

        OCR adapters can use this when a plate crop is present but cannot be
        read. The event carries no raw crop or guessed characters and is not a
        speed or access decision.
        """
        track = self.tracks.get(int(track_id))
        if track is None:
            return None
        return (
            "PLATE_READ_UNCERTAIN", f"VEHICLE_{track_id}", 0.0,
            f"Vehicle track {track_id} plate could not be read.",
            {
                "track_id": int(track_id),
                "reason": str(reason or "uncertain")[:120],
                "source_frame_id": source_frame_id,
                "entity_id": f"{str(camera_id or 'cam_0')}:vehicle:{int(track_id)}",
                "privacy_class": "restricted_local",
                "entity_type": "vehicle",
            },
        )

    def snapshot(self) -> dict:
        return {
            "capabilities": self.capability_state(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "active_tracks": [
                {"track_id": track.track_id, "class_name": track.class_name,
                 "center": track.center, "speed_mps": track.speed_mps,
                 "velocity": track.velocity,
                 "speed_kmh": round(track.speed_mps * 3.6, 2) if track.speed_mps is not None else None,
                 "source_frame_id": track.source_frame_id}
                for track in self.tracks.values()
            ],
        }
