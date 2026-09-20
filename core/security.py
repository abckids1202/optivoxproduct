"""Deterministic security signals built from tracked vision observations.

This module intentionally does not run a model. It turns tracks and detector
outputs into debounced, attributable security observations that can be linked
to the correlation and incident layers.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .policy_engine import PolicyEngine


def _finite_float(value: Any, default: float, minimum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    if minimum is not None:
        parsed = max(float(minimum), parsed)
    return parsed


@dataclass(frozen=True)
class ZoneRule:
    zone_id: str
    name: str
    bounds: Tuple[float, ...]
    shape: str = "rect"
    camera_id: Optional[str] = None
    restricted: bool = True
    allowed_names: Tuple[str, ...] = ()
    allowed_roles: Tuple[str, ...] = ()
    intrusion_enabled: bool = True
    exit_enabled: bool = True
    loitering_seconds: float = 15.0
    severity: int = 2
    active_weekdays: Tuple[int, ...] = ()
    active_start: Optional[str] = None
    active_end: Optional[str] = None


class SecuritySignalEngine:
    """Produce debounced intrusion, PPE, and crowd safety signals."""

    def __init__(self, cfg: Optional[dict] = None, policy_version: Optional[str] = None):
        root = cfg or {}
        self.cfg = root.get("SECURITY", {}) or {}
        policy = PolicyEngine(root).snapshot
        self.policy_version = str(policy_version or policy.policy_id)
        self.policy_issues = list(policy.issues)
        self.enabled = bool(self.cfg.get("ENABLED", True))
        self.configuration_issues: List[str] = []
        self.zones = self._parse_zones(
            self.cfg.get("ZONES") or root.get("SECURITY_ZONES") or [])
        # Security history is keyed by tracker generation, not only by the
        # numeric tracker ID. A reused ID must not inherit zone dwell, running
        # velocity, or intrusion cooldown state from the prior person.
        self._tracks: Dict[object, dict] = {}
        self._last_event: Dict[str, float] = {}
        self._evac_started: Dict[str, Optional[float]] = {}
        self._last_evac: Dict[str, Optional[float]] = {}
        self._ppe_available = bool(self.cfg.get("PPE", {}).get("AVAILABLE", False))
        self._ppe_enabled = bool(self.cfg.get("PPE", {}).get("ENABLED", False))
        self._ppe_missing_since: Dict[str, float] = {}
        self.roster_roles: Dict[str, str] = {}

    def set_roster_roles(self, roles: Optional[Dict[str, str]]) -> None:
        self.roster_roles = {
            str(name).casefold(): str(role).strip().casefold()
            for name, role in (roles or {}).items()
            if str(name).strip() and str(role or "").strip()
        }

    def _parse_zones(self, raw: Iterable[dict]) -> List[ZoneRule]:
        zones: List[ZoneRule] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                self.configuration_issues.append(f"zone_{index + 1}: zone must be an object")
                continue
            zone_id = str(item.get("id") or f"zone_{index + 1}").strip()
            if not zone_id:
                self.configuration_issues.append(f"zone_{index + 1}: zone id is required")
                continue
            zone_key = zone_id.casefold()
            if zone_key in seen_ids:
                self.configuration_issues.append(f"zone_{index + 1}: duplicate zone id '{zone_id}'")
                continue
            bounds = item.get("bounds") or item.get("coords") or item.get("rect")
            if not bounds or len(bounds) < 4:
                self.configuration_issues.append(f"zone_{index + 1}: at least four bounds are required")
                continue
            shape = str(item.get("shape", "rect")).lower()
            if shape not in {"rect", "polygon"}:
                self.configuration_issues.append(f"zone_{index + 1}: shape must be rect or polygon")
                continue
            if shape == "polygon" and len(bounds) < 6:
                self.configuration_issues.append(f"zone_{index + 1}: polygon requires at least three points")
                continue
            try:
                normalized_bounds = tuple(float(value) for value in bounds)
                if not all(math.isfinite(value) for value in normalized_bounds):
                    raise ValueError("zone bounds must be finite")
                schedule = item.get("schedule") or {}
                weekdays = schedule.get("weekdays") or item.get("active_weekdays") or []
                normalized_weekdays = tuple(int(value) for value in weekdays)
                if any(value < 0 or value > 6 for value in normalized_weekdays):
                    raise ValueError("zone weekdays must be between 0 and 6")
                for clock_value in (
                    schedule.get("start") or item.get("active_start"),
                    schedule.get("end") or item.get("active_end"),
                ):
                    if clock_value is None or str(clock_value).strip() == "":
                        continue
                    parts = str(clock_value).split(":", 1)
                    if len(parts) != 2:
                        raise ValueError("zone schedule must use HH:MM")
                    hour, minute = int(parts[0]), int(parts[1])
                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        raise ValueError("zone schedule is outside the 24-hour clock")
                zones.append(ZoneRule(
                    zone_id=zone_id,
                    name=str(item.get("name") or zone_id or f"Zone {index + 1}"),
                    bounds=normalized_bounds,
                    shape=shape,
                    camera_id=(str(item.get("camera_id") or "").strip() or None),
                    restricted=bool(item.get("restricted", True)),
                    allowed_names=tuple(str(value).casefold() for value in (item.get("allowed_names") or item.get("allowed") or [])),
                    allowed_roles=tuple(str(value).casefold() for value in (item.get("allowed_roles") or item.get("roles") or [])),
                    intrusion_enabled=bool(item.get("intrusion_enabled", True)),
                    exit_enabled=bool(item.get("exit_enabled", True)),
                    loitering_seconds=max(1.0, float(item.get("loitering_seconds", 15.0))),
                    severity=max(0, int(item.get("severity", 2))),
                    active_weekdays=normalized_weekdays,
                    active_start=(schedule.get("start") or item.get("active_start")),
                    active_end=(schedule.get("end") or item.get("active_end")),
                ))
            except (TypeError, ValueError, OverflowError):
                # A malformed zone must not crash the camera worker or become
                # an accidentally permissive rule. It is omitted and exposed
                # through capability_state for operator correction.
                self.configuration_issues.append(
                    f"zone_{index + 1}: invalid geometry, schedule, or threshold")
                continue
            seen_ids.add(zone_key)
        return zones

    def zones_snapshot(self) -> list[dict]:
        """Return a safe, editable representation of the active zone policy."""
        return [
            {
                "id": zone.zone_id,
                "name": zone.name,
                "bounds": list(zone.bounds),
                "shape": zone.shape,
                "camera_id": zone.camera_id,
                "restricted": zone.restricted,
                "allowed_names": list(zone.allowed_names),
                "allowed_roles": list(zone.allowed_roles),
                "intrusion_enabled": zone.intrusion_enabled,
                "exit_enabled": zone.exit_enabled,
                "loitering_seconds": zone.loitering_seconds,
                "severity": zone.severity,
                "schedule": {
                    "weekdays": list(zone.active_weekdays),
                    "start": zone.active_start,
                    "end": zone.active_end,
                },
            }
            for zone in self.zones
        ]

    def replace_zones(self, raw: Optional[Iterable[dict]]) -> dict:
        """Validate and atomically replace zone policy and dependent state."""
        if raw is None:
            raw = []
        if not isinstance(raw, (list, tuple)):
            return {"ok": False, "issues": ["zones must be a list"]}

        previous_zones = self.zones
        previous_issues = list(self.configuration_issues)
        self.configuration_issues = []
        parsed = self._parse_zones(raw)
        issues = list(self.configuration_issues)
        if issues or len(parsed) != len(raw):
            self.zones = previous_zones
            self.configuration_issues = previous_issues
            return {"ok": False, "issues": issues or ["zone policy contains invalid entries"]}

        self.zones = parsed
        self.configuration_issues = []
        # Geometry, dwell, and cooldown state cannot safely survive a policy
        # replacement because an existing track may now be in a new zone.
        self._tracks.clear()
        self._last_event.clear()
        self._evac_started.clear()
        self._last_evac.clear()
        self._ppe_missing_since.clear()
        self.cfg["ZONES"] = self.zones_snapshot()
        refreshed = PolicyEngine({"SECURITY": self.cfg}).snapshot
        self.policy_version = refreshed.policy_id
        self.policy_issues = list(refreshed.issues)
        return {"ok": True, "zones": self.zones_snapshot(), "count": len(self.zones)}

    def capability_state(self) -> dict:
        policy_valid = not self.policy_issues
        configuration_valid = not self.configuration_issues and policy_valid
        return {
            "configuration": "VALID" if configuration_valid else "INVALID",
            "policy_version": self.policy_version,
            "policy_valid": policy_valid,
            "intrusion": "AVAILABLE" if self.enabled and policy_valid and self.zones else "NOT_CONFIGURED",
            "loitering": "AVAILABLE" if self.enabled and policy_valid and self.zones else "NOT_CONFIGURED",
            "running": "AVAILABLE" if self.enabled and policy_valid else "DISABLED",
            "evacuation": "AVAILABLE" if self.enabled and policy_valid else "DISABLED",
            "ppe": "AVAILABLE" if self.enabled and policy_valid and self._ppe_enabled and self._ppe_available else "NOT_CONFIGURED",
        }

    @staticmethod
    def _inside(point: Tuple[float, float], zone: ZoneRule) -> bool:
        x, y = point
        if zone.shape == "polygon":
            inside = False
            points = list(zip(zone.bounds[::2], zone.bounds[1::2]))
            previous = points[-1]
            for current in points:
                denominator = previous[1] - current[1]
                if ((current[1] > y) != (previous[1] > y)
                        and abs(denominator) > 1e-9
                        and x < (previous[0] - current[0]) * (y - current[1]) /
                        denominator + current[0]):
                    inside = not inside
                previous = current
            return inside
        x1, y1, x2, y2 = zone.bounds[:4]
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    @staticmethod
    def _event_key(event_type: str, target: str, suffix: str = "") -> str:
        return f"{event_type}:{target}:{suffix}"

    @staticmethod
    def _zone_active(zone: ZoneRule, wallclock: Optional[str]) -> bool:
        """Evaluate an optional local schedule; absent schedules stay active."""
        if not zone.active_weekdays and not zone.active_start and not zone.active_end:
            return True
        if not wallclock:
            return True
        try:
            value = str(wallclock).replace("Z", "+00:00")
            current = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return True
        if zone.active_weekdays and current.weekday() not in zone.active_weekdays:
            return False
        current_minutes = current.hour * 60 + current.minute

        def parse_clock(value):
            if not value:
                return None
            hour, minute = str(value).split(":", 1)
            return int(hour) * 60 + int(minute)

        try:
            start = parse_clock(zone.active_start)
            end = parse_clock(zone.active_end)
        except (TypeError, ValueError):
            return True
        if start is None and end is None:
            return True
        if start is None:
            return current_minutes <= end
        if end is None:
            return current_minutes >= start
        if start <= end:
            return start <= current_minutes <= end
        # Overnight window, e.g. 22:00 through 06:00.
        return current_minutes >= start or current_minutes <= end

    def _emit(self, events: list, event_type: str, target: str, confidence: float,
              details: str, metadata: dict, now: float, cooldown: float = 10.0) -> None:
        metadata.setdefault("policy_version", self.policy_version)
        key = self._event_key(
            event_type,
            target,
            ":".join(str(metadata.get(name, "")) for name in (
                "camera_id", "zone_id", "track_generation")))
        last_event = self._last_event.get(key)
        if last_event is not None and now - last_event < cooldown:
            return
        self._last_event[key] = now
        events.append((event_type, target, float(confidence), details, metadata))

    def _identity_for(self, track_id: int, faces: Dict[int, dict]) -> tuple[str, str, str]:
        face = faces.get(track_id) or {}
        state = str(face.get("identity_state") or "UNRESOLVED").upper()
        name = str(face.get("name") or "UNKNOWN").strip()
        role = str(face.get("role") or "").strip().casefold()
        if not role and name:
            role = self.roster_roles.get(name.casefold(), "")
        if state != "CONFIRMED" or not name or name in {"UNKNOWN", "SPOOF"}:
            return "UNKNOWN", state, role
        return name, state, role

    def _zone_events(self, events: list, track_id: int, center: Tuple[float, float],
                     face: dict, now: float, wallclock: Optional[str] = None,
                     state_key: object = None, camera_id: str = "cam_0") -> None:
        state = self._tracks.setdefault(
            state_key if state_key is not None else track_id,
            {"previous": center, "last_at": now, "zones": {}})
        identity, identity_state, role = self._identity_for(track_id, {track_id: face})
        try:
            generation = int(face.get("track_generation") or (
                state_key[2] if isinstance(state_key, tuple) and len(state_key) > 2 else 1))
        except (TypeError, ValueError, OverflowError):
            generation = 1
        allowed_name = identity.casefold()
        for zone in self.zones:
            if zone.camera_id and zone.camera_id != camera_id:
                continue
            if not self._zone_active(zone, wallclock):
                previous = state["zones"].setdefault(zone.zone_id, {
                    "inside": False, "entered_at": None, "loitered": False,
                    "initialized": False,
                })
                previous.update({
                    "inside": False, "entered_at": None, "loitered": False,
                    "initialized": False,
                })
                continue
            inside = self._inside(center, zone)
            previous = state["zones"].setdefault(zone.zone_id, {
                "inside": False, "entered_at": None, "loitered": False,
                "initialized": False,
            })
            if not previous.get("initialized", False):
                # The first observation establishes a baseline. It is not
                # evidence of an outside-to-inside transition because the
                # process may have started after the person entered.
                previous.update({
                    "inside": inside,
                    "entered_at": now if inside else None,
                    "loitered": False,
                    "initialized": True,
                })
                continue
            metadata = {
                "track_id": track_id,
                "camera_id": camera_id,
                "zone_id": zone.zone_id,
                "zone_name": zone.name,
                "identity_state": identity_state,
                "role": role or None,
                "track_generation": generation,
            }
            if inside and not previous["inside"]:
                previous["entered_at"] = now
                previous["loitered"] = False
                self._emit(
                    events, "ZONE_ENTRY", identity if identity != "UNKNOWN" else f"ID_{track_id}",
                    1.0, f"Track {track_id} entered {zone.name}.", metadata, now, 5.0)
                is_allowed = (
                    not zone.restricted
                    or (bool(zone.allowed_names) and allowed_name in zone.allowed_names)
                    or (bool(zone.allowed_roles) and role in zone.allowed_roles)
                )
                if zone.intrusion_enabled and not is_allowed:
                    self._emit(
                        events, "ZONE_INTRUSION", identity if identity != "UNKNOWN" else f"ID_{track_id}",
                        0.95, f"Unauthorized presence entered {zone.name}.",
                        {**metadata, "severity": zone.severity}, now, 20.0)
            elif not inside and previous["inside"]:
                if zone.exit_enabled:
                    self._emit(
                        events, "ZONE_EXIT", identity if identity != "UNKNOWN" else f"ID_{track_id}",
                        1.0, f"Track {track_id} exited {zone.name}.", metadata, now, 5.0)
                previous["entered_at"] = None
                previous["loitered"] = False
            if inside and previous["entered_at"] is not None:
                dwell = now - previous["entered_at"]
                if dwell >= zone.loitering_seconds and not previous["loitered"]:
                    previous["loitered"] = True
                    self._emit(
                        events, "LOITERING", identity if identity != "UNKNOWN" else f"ID_{track_id}",
                        0.85, f"Track {track_id} remained in {zone.name} for {dwell:.1f}s.",
                        {**metadata, "dwell_seconds": round(dwell, 2)}, now, 30.0)
            previous["inside"] = inside

    def _running_event(self, events: list, track_id: int, center: Tuple[float, float], now: float,
                       state_key: object = None, camera_id: str = "cam_0",
                       frame_size: Optional[Tuple[float, float]] = None) -> None:
        state = self._tracks.setdefault(
            state_key if state_key is not None else track_id,
            {"previous": center, "last_at": now, "zones": {}})
        try:
            generation = int(
                state_key[2] if isinstance(state_key, tuple) and len(state_key) > 2 else 1)
        except (TypeError, ValueError, OverflowError):
            generation = 1
        elapsed = max(1e-6, now - float(state.get("last_at", now)))
        previous = state.get("previous") or center
        speed = math.hypot(center[0] - previous[0], center[1] - previous[1]) / elapsed
        history = state.setdefault("speeds", deque(maxlen=8))
        history.append(speed)
        running_cfg = self.cfg.get("RUNNING", {}) or {}
        pixel_threshold = _finite_float(
            running_cfg.get("SPEED_THRESHOLD_PX_SEC", 280.0), 280.0, 0.0)
        diagonal = None
        try:
            width, height = float(frame_size[0]), float(frame_size[1])
            diagonal = math.hypot(width, height)
            if not math.isfinite(diagonal) or diagonal <= 0:
                diagonal = None
        except (TypeError, ValueError, IndexError, OverflowError):
            diagonal = None
        # A normalized threshold keeps the detector's behavior stable when
        # the processing resolution changes. The explicit normalized setting
        # is preferred; otherwise the legacy pixel threshold is scaled from a
        # documented reference diagonal. Without frame dimensions we retain
        # the legacy pixel path for compatibility and expose that fact below.
        reference_diagonal = _finite_float(
            running_cfg.get("REFERENCE_DIAGONAL_PX", 640.0), 640.0, 1.0)
        configured_normalized = running_cfg.get("SPEED_THRESHOLD_NORMALIZED")
        if diagonal is not None:
            if configured_normalized is not None:
                threshold = _finite_float(configured_normalized, pixel_threshold / reference_diagonal, 0.0)
            else:
                threshold = pixel_threshold / reference_diagonal
            measured_speed = speed / diagonal
            threshold_mode = "frame_normalized"
        else:
            threshold = pixel_threshold
            measured_speed = speed
            threshold_mode = "pixel_fallback"
        confirm_sec = _finite_float(
            self.cfg.get("RUNNING", {}).get("CONFIRM_SECONDS", 0.6),
            0.6, 0.0)
        average = sum(history) / len(history)
        if diagonal is not None:
            normalized_history = state.setdefault("normalized_speeds", deque(maxlen=8))
            normalized_history.append(measured_speed)
            average = sum(normalized_history) / len(normalized_history)
        if average >= threshold:
            state["running_since"] = state.get("running_since") or now
            if now - state["running_since"] >= confirm_sec:
                self._emit(
                    events, "RUNNING", f"ID_{track_id}", min(1.0, average / max(threshold, 1.0)),
                    (f"Track {track_id} sustained {speed:.1f}px/s movement."
                     if diagonal is None else
                     f"Track {track_id} sustained normalized movement."),
                    {"track_id": track_id, "camera_id": camera_id,
                     "track_generation": generation,
                     "speed_px_sec": round(speed, 2),
                     "speed_normalized": round(average, 6) if diagonal is not None else None,
                     "threshold": round(threshold, 6) if diagonal is not None else round(threshold, 2),
                     "threshold_mode": threshold_mode,
                     "frame_diagonal_px": round(diagonal, 2) if diagonal is not None else None},
                    now, 20.0)
        else:
            state["running_since"] = None
        state["previous"] = center
        state["last_at"] = now

    def _ppe_events(self, events: list, object_detections: list,
                    tracked: Dict[int, Tuple[float, float]], faces: Dict[int, dict], now: float,
                    camera_id: str = "cam_0",
                    track_generations: Optional[Dict[int, int]] = None) -> None:
        if not (self.enabled and self._ppe_enabled and self._ppe_available):
            return
        ppe_cfg = self.cfg.get("PPE", {})
        required = {str(value).casefold() for value in (ppe_cfg.get("REQUIRED_OBJECTS") or {"helmet", "mask", "vest"})}
        min_conf = _finite_float(ppe_cfg.get("MIN_CONFIDENCE", 0.55), 0.55, 0.0)
        overlap_threshold = _finite_float(ppe_cfg.get("MIN_OVERLAP", 0.15), 0.15, 0.0)
        persistence = _finite_float(ppe_cfg.get("PERSISTENCE_SECONDS", 1.0), 1.0, 0.1)
        track_generations = track_generations or {}
        active_keys = set()
        people = [item for item in object_detections
                  if isinstance(item, dict)
                  and str(item.get("class_name", "")).casefold() == "person"]
        ppe = []
        for item in object_detections:
            if not isinstance(item, dict) or str(item.get("class_name", "")).casefold() not in required:
                continue
            if _finite_float(item.get("confidence", 0.0), 0.0, 0.0) >= min_conf:
                ppe.append(item)
        for index, person in enumerate(people):
            try:
                box = person.get("bbox") or (0, 0, 0, 0)
                px1, py1, px2, py2 = [float(value) for value in box]
                if not all(math.isfinite(value) for value in (px1, py1, px2, py2)):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                # Ignore malformed detector rows without stopping unrelated
                # security signals in the same frame.
                continue
            area = max(1.0, (px2 - px1) * (py2 - py1))
            found = set()
            for item in ppe:
                try:
                    bx1, by1, bx2, by2 = [float(value) for value in item.get("bbox") or (0, 0, 0, 0)]
                    if not all(math.isfinite(value) for value in (bx1, by1, bx2, by2)):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    continue
                ix1, iy1 = max(px1, bx1), max(py1, by1)
                ix2, iy2 = min(px2, bx2), min(py2, by2)
                overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / area
                if overlap >= overlap_threshold:
                    found.add(str(item.get("class_name", "")).casefold())
            missing = sorted(required - found)
            person_center = ((px1 + px2) * 0.5, (py1 + py2) * 0.5)
            track_id = None
            if tracked:
                track_id = min(
                    tracked,
                    key=lambda candidate: math.hypot(
                        float(tracked[candidate][0]) - person_center[0],
                        float(tracked[candidate][1]) - person_center[1],
                    ),
                )
                if math.hypot(
                    float(tracked[track_id][0]) - person_center[0],
                    float(tracked[track_id][1]) - person_center[1],
                ) > max(px2 - px1, py2 - py1, 120.0):
                    track_id = None
            target = f"ID_{track_id}" if track_id is not None else f"PERSON_{index + 1}"
            try:
                generation = int(
                    (faces.get(int(track_id), {}) if track_id is not None else {}).get(
                        "track_generation")
                    or (track_generations.get(int(track_id), 1) if track_id is not None else 1)
                    or 1
                )
            except (TypeError, ValueError, OverflowError):
                generation = 1
            # A numeric track ID can be reused. Include its generation so
            # missing-PPE dwell cannot carry over from a previous person.
            ppe_key = f"{camera_id}:{target}:generation-{generation}:{index}"
            active_keys.add(ppe_key)
            if missing:
                self._ppe_missing_since.setdefault(ppe_key, now)
                if now - self._ppe_missing_since[ppe_key] >= persistence:
                    identity_state = str((faces.get(int(track_id), {}) if track_id is not None else {}).get(
                        "identity_state", "UNRESOLVED"))
                    self._emit(
                        events, "PPE_VIOLATION", target, 0.8,
                        f"Required PPE missing: {', '.join(missing)}.",
                        {"missing": missing, "required": sorted(required),
                         "track_id": track_id, "camera_id": camera_id,
                         "track_generation": generation,
                         "identity_state": identity_state,
                         "persistence_seconds": round(now - self._ppe_missing_since[ppe_key], 2)},
                        now, _finite_float(ppe_cfg.get("COOLDOWN_SECONDS", 15.0), 15.0, 0.0))
            else:
                self._ppe_missing_since.pop(ppe_key, None)
        prefix = f"{camera_id}:"
        for key in list(self._ppe_missing_since):
            if key.startswith(prefix) and key not in active_keys:
                self._ppe_missing_since.pop(key, None)

    def _evacuation_event(self, events: list, tracked: Dict[int, Tuple[float, float]], now: float,
                          state_keys: Optional[Dict[int, object]] = None,
                          camera_id: str = "cam_0",
                          frame_size: Optional[Tuple[float, float]] = None) -> None:
        cfg = self.cfg.get("EVACUATION", {})
        try:
            minimum = max(1, int(cfg.get("MIN_PEOPLE", 4)))
        except (TypeError, ValueError, OverflowError):
            minimum = 4
        pixel_threshold = _finite_float(
            cfg.get("AVERAGE_SPEED_THRESHOLD_PX_SEC", 25.0), 25.0, 0.0)
        diagonal = None
        try:
            width, height = float(frame_size[0]), float(frame_size[1])
            diagonal = math.hypot(width, height)
            if not math.isfinite(diagonal) or diagonal <= 0:
                diagonal = None
        except (TypeError, ValueError, IndexError, OverflowError):
            diagonal = None
        reference_diagonal = _finite_float(
            cfg.get("REFERENCE_DIAGONAL_PX", 640.0), 640.0, 1.0)
        configured_normalized = cfg.get("AVERAGE_SPEED_THRESHOLD_NORMALIZED")
        if diagonal is not None:
            threshold = _finite_float(
                configured_normalized,
                pixel_threshold / reference_diagonal,
                0.0,
            ) if configured_normalized is not None else pixel_threshold / reference_diagonal
            threshold_mode = "frame_normalized"
        else:
            threshold = pixel_threshold
            threshold_mode = "pixel_fallback"
        if len(tracked) < minimum:
            self._evac_started[str(camera_id)] = None
            return
        state_keys = state_keys or {}
        speeds = [sum(self._tracks.get(state_keys.get(int(track_id), track_id), {}).get("speeds", ())) /
                  max(1, len(self._tracks.get(state_keys.get(int(track_id), track_id), {}).get("speeds", ())))
                  for track_id in tracked]
        average_px = sum(speeds) / max(1, len(speeds))
        average = average_px / diagonal if diagonal is not None else average_px
        if average >= threshold:
            camera_key = str(camera_id)
            self._evac_started[camera_key] = self._evac_started.get(camera_key) or now
            confirm_seconds = _finite_float(
                cfg.get("CONFIRM_SECONDS", 1.0), 1.0, 0.0)
            cooldown_seconds = _finite_float(
                cfg.get("COOLDOWN_SECONDS", 30.0), 30.0, 0.0)
            last_evac = self._last_evac.get(camera_key)
            if (now - self._evac_started[camera_key] >= confirm_seconds
                    and (last_evac is None or now - last_evac >= cooldown_seconds)):
                self._last_evac[camera_key] = now
                self._emit(
                    events, "EVACUATION_ALERT", "SYSTEM", 0.9,
                    (f"{len(tracked)} people moving at {average_px:.1f}px/s."
                     if diagonal is None else
                     f"{len(tracked)} people moving at normalized speed."),
                    {"people": len(tracked), "camera_id": camera_id,
                     "average_speed_px_sec": round(average_px, 2),
                     "average_speed_normalized": round(average, 6) if diagonal is not None else None,
                     "threshold": round(threshold, 6) if diagonal is not None else round(threshold, 2),
                     "threshold_mode": threshold_mode,
                     "frame_diagonal_px": round(diagonal, 2) if diagonal is not None else None}, now, 1.0)
        else:
            self._evac_started[str(camera_id)] = None

    def update(self, tracked: Dict[int, Tuple[float, float]], faces_info: Optional[Iterable[dict]] = None,
               object_detections: Optional[Iterable[dict]] = None, now: Optional[float] = None,
               wallclock: Optional[str] = None,
               track_generations: Optional[Dict[int, int]] = None,
               camera_id: str = "cam_0",
               frame_size: Optional[Tuple[float, float]] = None) -> list[tuple]:
        # Capability state and execution must agree. A malformed active
        # policy must not continue producing alerts merely because a subset of
        # its parsed rules survived validation.
        if not self.enabled or self.policy_issues or self.configuration_issues:
            return []
        current = time.monotonic() if now is None else float(now)
        events: list[tuple] = []
        faces = {int(item.get("oid")): item for item in (faces_info or ()) if item.get("oid") is not None}
        track_generations = track_generations or {}
        active_state_keys: Dict[int, object] = {}
        for track_id, center in tracked.items():
            center = (float(center[0]), float(center[1]))
            track_id = int(track_id)
            face = faces.get(track_id, {})
            try:
                generation = int(face.get("track_generation") or track_generations.get(track_id) or 1)
            except (TypeError, ValueError):
                generation = 1
            state_key = (str(camera_id), track_id, generation)
            active_state_keys[track_id] = state_key
            self._running_event(events, track_id, center, current, state_key=state_key,
                                camera_id=str(camera_id), frame_size=frame_size)
            self._zone_events(events, track_id, center, face, current, wallclock,
                              state_key=state_key, camera_id=str(camera_id))
        self._ppe_events(
            events, list(object_detections or ()), tracked, faces, current,
            camera_id=str(camera_id), track_generations=track_generations)
        self._evacuation_event(events, tracked, current, active_state_keys,
                               camera_id=str(camera_id), frame_size=frame_size)
        active_ids = {int(track_id) for track_id in tracked}
        for state_key in list(self._tracks):
            if isinstance(state_key, tuple):
                track_id = state_key[1] if len(state_key) >= 2 else state_key[0]
            else:
                track_id = state_key
            if track_id not in active_ids and current - self._tracks[state_key].get("last_at", current) > 15.0:
                self._tracks.pop(state_key, None)
        return events
