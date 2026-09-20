"""Versioned, deterministic policy snapshots for OptiVox decisions.

Perception code should not own operational policy. This module produces a
small immutable contract that can be attached to attendance, security, and
replay decisions. It deliberately does not execute models or write data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _number(value: Any, name: str, issues: list[str], *, minimum: float = 0.0) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        issues.append(f"{name} must be numeric")
        return
    if not math.isfinite(parsed) or parsed < minimum:
        issues.append(f"{name} must be finite and >= {minimum:g}")


@dataclass(frozen=True)
class PolicySnapshot:
    """An immutable, content-addressed policy contract."""

    policy_id: str
    declared_version: str
    organization_id: str
    site_id: str
    device_id: str
    document: dict[str, Any]
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def metadata(self) -> dict[str, str]:
        return {
            "policy_version": self.policy_id,
            "policy_declared_version": self.declared_version,
            "policy_valid": "true" if self.valid else "false",
        }

    def attendance_school_day(self, local_now: datetime) -> bool:
        if not self.valid:
            return False
        attendance = self.document.get("attendance", {})
        days = set(int(item) for item in attendance.get("school_days", []))
        holidays = set(str(item) for item in attendance.get("holidays", []))
        return local_now.weekday() in days and local_now.date().isoformat() not in holidays


class PolicyEngine:
    """Build and validate one policy snapshot for a deployment boundary."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        organization_id: str = "",
        site_id: str = "",
        device_id: str = "",
    ):
        root = dict(config or {})
        declared = str(
            root.get("POLICY_VERSION")
            or (root.get("POLICY") or {}).get("VERSION")
            or "1"
        ).strip() or "1"
        attendance = dict(root.get("ATTENDANCE") or {})
        security = dict(root.get("SECURITY") or {})
        document = {
            "attendance": {
                "enabled": bool(attendance.get("ENABLED", True)),
                "school_days": list(attendance.get("SCHOOL_DAYS", [0, 1, 2, 3, 4])),
                "holidays": list(attendance.get("HOLIDAYS", [])),
                "late_grace_min": attendance.get("LATE_GRACE_MIN", 0),
                "auto_clockout_timeout_min": attendance.get("AUTO_CLOCKOUT_TIMEOUT_MIN", 15),
                "presence_session_close_after_sec": attendance.get("PRESENCE_SESSION_CLOSE_AFTER_SEC", 10),
                "center_mode_require_liveness": bool(attendance.get("CENTER_MODE_REQUIRE_LIVENESS", True)),
            },
            "security": {
                "enabled": bool(security.get("ENABLED", True)),
                "zones": _json_safe(security.get("ZONES") or []),
                "running": _json_safe(security.get("RUNNING") or {}),
                "evacuation": _json_safe(security.get("EVACUATION") or {}),
                "ppe": _json_safe(security.get("PPE") or {}),
            },
            "correlation": _json_safe(root.get("CORRELATION_CORE") or {}),
            "notifications": _json_safe(root.get("NOTIFICATIONS") or {}),
            "privacy": _json_safe(root.get("PRIVACY") or {}),
        }
        issues = self._validate(document)
        canonical = json.dumps(_json_safe(document), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        self.snapshot = PolicySnapshot(
            policy_id=f"{declared}:{digest}",
            declared_version=declared,
            organization_id=str(organization_id or ""),
            site_id=str(site_id or ""),
            device_id=str(device_id or ""),
            document=document,
            issues=tuple(issues),
        )

    @staticmethod
    def _validate(document: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        attendance = document["attendance"]
        days = attendance.get("school_days")
        if not isinstance(days, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 6 for item in days):
            issues.append("attendance.school_days must contain integers from 0 to 6")
        elif len(set(days)) != len(days):
            issues.append("attendance.school_days must not contain duplicates")
        holidays = attendance.get("holidays")
        if not isinstance(holidays, list):
            issues.append("attendance.holidays must be a list")
        else:
            for item in holidays:
                try:
                    date.fromisoformat(str(item))
                except (TypeError, ValueError):
                    issues.append(f"attendance holiday is not ISO formatted: {item}")
        for key in ("late_grace_min", "auto_clockout_timeout_min", "presence_session_close_after_sec"):
            _number(attendance.get(key), f"attendance.{key}", issues)
        security = document["security"]
        zones = security.get("zones")
        if not isinstance(zones, list):
            issues.append("security.zones must be a list")
        else:
            seen_zone_ids: set[str] = set()
            for index, zone in enumerate(zones):
                prefix = f"security.zones[{index}]"
                if not isinstance(zone, dict):
                    issues.append(f"{prefix} must be an object")
                    continue
                zone_id = str(zone.get("id") or "").strip().casefold()
                if not zone_id:
                    issues.append(f"{prefix}.id is required")
                elif zone_id in seen_zone_ids:
                    issues.append(f"{prefix}.id must be unique")
                seen_zone_ids.add(zone_id)
                shape = str(zone.get("shape", "rect")).casefold()
                bounds = zone.get("bounds") or zone.get("coords") or zone.get("rect")
                if shape not in {"rect", "polygon"}:
                    issues.append(f"{prefix}.shape must be rect or polygon")
                if not isinstance(bounds, (list, tuple)):
                    issues.append(f"{prefix}.bounds must be a list")
                else:
                    minimum = 6 if shape == "polygon" else 4
                    if len(bounds) < minimum:
                        issues.append(f"{prefix}.bounds has too few coordinates")
                    try:
                        values = [float(value) for value in bounds]
                        if not all(math.isfinite(value) for value in values):
                            raise ValueError
                    except (TypeError, ValueError, OverflowError):
                        issues.append(f"{prefix}.bounds must contain finite numbers")
                schedule = zone.get("schedule") or {}
                if not isinstance(schedule, dict):
                    issues.append(f"{prefix}.schedule must be an object")
                else:
                    weekdays = schedule.get("weekdays", zone.get("active_weekdays", [])) or []
                    if (not isinstance(weekdays, (list, tuple)) or
                            any(isinstance(day, bool) or not isinstance(day, int) or day < 0 or day > 6
                                for day in weekdays)):
                        issues.append(f"{prefix}.schedule.weekdays must contain integers from 0 to 6")
                    for clock_name in ("start", "end"):
                        clock_value = schedule.get(clock_name, zone.get(f"active_{clock_name}"))
                        if clock_value in (None, ""):
                            continue
                        try:
                            hour, minute = str(clock_value).split(":", 1)
                            if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
                                raise ValueError
                        except (TypeError, ValueError, OverflowError):
                            issues.append(f"{prefix}.schedule.{clock_name} must use HH:MM")
                _number(zone.get("severity", 2), f"{prefix}.severity", issues)
                _number(zone.get("loitering_seconds", 15), f"{prefix}.loitering_seconds", issues, minimum=1.0)

        running = security.get("running") or {}
        evacuation = security.get("evacuation") or {}
        if not isinstance(running, dict):
            issues.append("security.running must be an object")
        else:
            _number(running.get("SPEED_THRESHOLD_PX_SEC", 0), "security.running.speed_threshold", issues, minimum=0.0)
            _number(running.get("CONFIRM_SECONDS", 0), "security.running.confirm_seconds", issues, minimum=0.0)
        if not isinstance(evacuation, dict):
            issues.append("security.evacuation must be an object")
        else:
            _number(evacuation.get("MIN_PEOPLE", 1), "security.evacuation.min_people", issues, minimum=1.0)
            _number(evacuation.get("AVERAGE_SPEED_THRESHOLD_PX_SEC", 0), "security.evacuation.average_speed_threshold", issues, minimum=0.0)
            _number(evacuation.get("CONFIRM_SECONDS", 0), "security.evacuation.confirm_seconds", issues, minimum=0.0)
        ppe = security.get("ppe") or {}
        if not isinstance(ppe, dict):
            issues.append("security.ppe must be an object")
        else:
            for key in ("MIN_CONFIDENCE", "MIN_OVERLAP"):
                try:
                    value = float(ppe.get(key, 0.0))
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    issues.append(f"security.ppe.{key.lower()} must be between 0 and 1")
        return issues
