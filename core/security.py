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


@dataclass(frozen=True)
class ZoneRule:
    zone_id: str
    name: str
    bounds: Tuple[float, ...]
    shape: str = "rect"
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

    def __init__(self, cfg: Optional[dict] = None):
        root = cfg or {}
        self.cfg = root.get("SECURITY", {}) or {}
        self.enabled = bool(self.cfg.get("ENABLED", True))
        self.zones = self._parse_zones(self.cfg.get("ZONES") or root.get("SECURITY_ZONES") or [])
        self._tracks: Dict[int, dict] = {}
        self._last_event: Dict[str, float] = {}
        self._evac_started: Optional[float] = None
        self._last_evac: Optional[float] = None
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

    @staticmethod
    def _parse_zones(raw: Iterable[dict]) -> List[ZoneRule]:
        zones: List[ZoneRule] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            bounds = item.get("bounds") or item.get("coords") or item.get("rect")
            if not bounds or len(bounds) < 4:
                continue
            shape = str(item.get("shape", "rect")).lower()
            if shape == "polygon" and len(bounds) < 6:
                continue
            zones.append(ZoneRule(
                zone_id=str(item.get("id") or f"zone_{index + 1}"),
                name=str(item.get("name") or item.get("id") or f"Zone {index + 1}"),
                bounds=tuple(float(value) for value in bounds),
                shape=shape,
                restricted=bool(item.get("restricted", True)),
                allowed_names=tuple(str(value).casefold() for value in (item.get("allowed_names") or item.get("allowed") or [])),
                allowed_roles=tuple(str(value).casefold() for value in (item.get("allowed_roles") or item.get("roles") or [])),
                intrusion_enabled=bool(item.get("intrusion_enabled", True)),
                exit_enabled=bool(item.get("exit_enabled", True)),
                loitering_seconds=max(1.0, float(item.get("loitering_seconds", 15.0))),
                severity=max(0, int(item.get("severity", 2))),
                active_weekdays=tuple(int(value) for value in ((item.get("schedule") or {}).get("weekdays") or item.get("active_weekdays") or [])),
                active_start=((item.get("schedule") or {}).get("start") or item.get("active_start")),
                active_end=((item.get("schedule") or {}).get("end") or item.get("active_end")),
            ))
        return zones

    def capability_state(self) -> dict:
        return {
            "intrusion": "AVAILABLE" if self.enabled and self.zones else "NOT_CONFIGURED",
            "loitering": "AVAILABLE" if self.enabled and self.zones else "NOT_CONFIGURED",
            "running": "AVAILABLE" if self.enabled else "DISABLED",
            "evacuation": "AVAILABLE" if self.enabled else "DISABLED",
            "ppe": "AVAILABLE" if self.enabled and self._ppe_enabled and self._ppe_available else "NOT_CONFIGURED",
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
        key = self._event_key(event_type, target, str(metadata.get("zone_id", "")))
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
                     face: dict, now: float, wallclock: Optional[str] = None) -> None:
        state = self._tracks.setdefault(track_id, {"previous": center, "last_at": now, "zones": {}})
        identity, identity_state, role = self._identity_for(track_id, {track_id: face})
        allowed_name = identity.casefold()
        for zone in self.zones:
            if not self._zone_active(zone, wallclock):
                previous = state["zones"].setdefault(zone.zone_id, {
                    "inside": False, "entered_at": None, "loitered": False,
                })
                previous.update({"inside": False, "entered_at": None, "loitered": False})
                continue
            inside = self._inside(center, zone)
            previous = state["zones"].setdefault(zone.zone_id, {
                "inside": False, "entered_at": None, "loitered": False,
            })
            metadata = {
                "track_id": track_id,
                "zone_id": zone.zone_id,
                "zone_name": zone.name,
                "identity_state": identity_state,
                "role": role or None,
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

    def _running_event(self, events: list, track_id: int, center: Tuple[float, float], now: float) -> None:
        state = self._tracks.setdefault(track_id, {"previous": center, "last_at": now, "zones": {}})
        elapsed = max(1e-6, now - float(state.get("last_at", now)))
        previous = state.get("previous") or center
        speed = math.hypot(center[0] - previous[0], center[1] - previous[1]) / elapsed
        history = state.setdefault("speeds", deque(maxlen=8))
        history.append(speed)
        threshold = float(self.cfg.get("RUNNING", {}).get("SPEED_THRESHOLD_PX_SEC", 280.0))
        confirm_sec = float(self.cfg.get("RUNNING", {}).get("CONFIRM_SECONDS", 0.6))
        average = sum(history) / len(history)
        if average >= threshold:
            state["running_since"] = state.get("running_since") or now
            if now - state["running_since"] >= confirm_sec:
                self._emit(
                    events, "RUNNING", f"ID_{track_id}", min(1.0, average / max(threshold, 1.0)),
                    f"Track {track_id} sustained {average:.1f}px/s movement.",
                    {"track_id": track_id, "speed_px_sec": round(average, 2)}, now, 20.0)
        else:
            state["running_since"] = None
        state["previous"] = center
        state["last_at"] = now

    def _ppe_events(self, events: list, object_detections: list,
                    tracked: Dict[int, Tuple[float, float]], faces: Dict[int, dict], now: float) -> None:
        if not (self.enabled and self._ppe_enabled and self._ppe_available):
            return
        ppe_cfg = self.cfg.get("PPE", {})
        required = {str(value).casefold() for value in (ppe_cfg.get("REQUIRED_OBJECTS") or {"helmet", "mask", "vest"})}
        min_conf = float(ppe_cfg.get("MIN_CONFIDENCE", 0.55))
        overlap_threshold = float(ppe_cfg.get("MIN_OVERLAP", 0.15))
        persistence = max(0.1, float(ppe_cfg.get("PERSISTENCE_SECONDS", 1.0)))
        people = [item for item in object_detections if str(item.get("class_name", "")).casefold() == "person"]
        ppe = [item for item in object_detections if str(item.get("class_name", "")).casefold() in required and float(item.get("confidence", 0.0)) >= min_conf]
        for index, person in enumerate(people):
            box = person.get("bbox") or (0, 0, 0, 0)
            px1, py1, px2, py2 = [float(value) for value in box]
            area = max(1.0, (px2 - px1) * (py2 - py1))
            found = set()
            for item in ppe:
                bx1, by1, bx2, by2 = [float(value) for value in item.get("bbox") or (0, 0, 0, 0)]
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
            ppe_key = f"{target}:{index}"
            if missing:
                self._ppe_missing_since.setdefault(ppe_key, now)
                if now - self._ppe_missing_since[ppe_key] >= persistence:
                    identity_state = str((faces.get(int(track_id), {}) if track_id is not None else {}).get(
                        "identity_state", "UNRESOLVED"))
                    self._emit(
                        events, "PPE_VIOLATION", target, 0.8,
                        f"Required PPE missing: {', '.join(missing)}.",
                        {"missing": missing, "required": sorted(required),
                         "track_id": track_id, "identity_state": identity_state,
                         "persistence_seconds": round(now - self._ppe_missing_since[ppe_key], 2)},
                        now, float(ppe_cfg.get("COOLDOWN_SECONDS", 15.0)))
            else:
                self._ppe_missing_since.pop(ppe_key, None)

    def _evacuation_event(self, events: list, tracked: Dict[int, Tuple[float, float]], now: float) -> None:
        cfg = self.cfg.get("EVACUATION", {})
        minimum = int(cfg.get("MIN_PEOPLE", 4))
        threshold = float(cfg.get("AVERAGE_SPEED_THRESHOLD_PX_SEC", 25.0))
        if len(tracked) < minimum:
            self._evac_started = None
            return
        speeds = [sum(self._tracks.get(track_id, {}).get("speeds", ())) /
                  max(1, len(self._tracks.get(track_id, {}).get("speeds", ())))
                  for track_id in tracked]
        average = sum(speeds) / max(1, len(speeds))
        if average >= threshold:
            self._evac_started = self._evac_started or now
            if (now - self._evac_started >= float(cfg.get("CONFIRM_SECONDS", 1.0))
                    and (self._last_evac is None or
                         now - self._last_evac >= float(cfg.get("COOLDOWN_SECONDS", 30.0)))):
                self._last_evac = now
                self._emit(
                    events, "EVACUATION_ALERT", "SYSTEM", 0.9,
                    f"{len(tracked)} people moving at {average:.1f}px/s.",
                    {"people": len(tracked), "average_speed_px_sec": round(average, 2)}, now, 1.0)
        else:
            self._evac_started = None

    def update(self, tracked: Dict[int, Tuple[float, float]], faces_info: Optional[Iterable[dict]] = None,
               object_detections: Optional[Iterable[dict]] = None, now: Optional[float] = None,
               wallclock: Optional[str] = None) -> list[tuple]:
        if not self.enabled:
            return []
        current = time.monotonic() if now is None else float(now)
        events: list[tuple] = []
        faces = {int(item.get("oid")): item for item in (faces_info or ()) if item.get("oid") is not None}
        for track_id, center in tracked.items():
            center = (float(center[0]), float(center[1]))
            self._running_event(events, int(track_id), center, current)
            self._zone_events(events, int(track_id), center, faces.get(int(track_id), {}), current, wallclock)
        self._ppe_events(events, list(object_detections or ()), tracked, faces, current)
        self._evacuation_event(events, tracked, current)
        for track_id in list(self._tracks):
            if track_id not in tracked and current - self._tracks[track_id].get("last_at", current) > 15.0:
                self._tracks.pop(track_id, None)
        return events
