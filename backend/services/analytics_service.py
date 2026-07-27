from __future__ import annotations

from ..database import fetch_all, fetch_one
from .attendance_service import attendance_summary
from .runtime_service import live_state


def overview() -> dict:
    live = live_state()
    people = fetch_one("select count(*) as c from people") or {"c": 0}
    alerts = fetch_one("select count(*) as c from alert_log where status='sent'") or {"c": 0}
    summary = attendance_summary()
    return {
        "registered_people": people["c"],
        "present_today": summary["present"],
        "currently_visible": live["summary"]["visibleNow"],
        "unknown_detections_today": live["summary"]["unknownToday"],
        "security_events_today": live["summary"]["securityEvents"],
        "alerts_sent": alerts["c"],
        "current_fps": live["engine"]["fps"],
    }


def attendance(days: int = 7) -> dict:
    by_day = fetch_all(
        """
        select date as day,
               count(*) as present,
               sum(case when late_minutes > 0 then 1 else 0 end) as late
        from attendance
        group by date order by date desc limit ?
        """,
        [days],
    )
    return {"attendanceByDay": list(reversed(by_day)), "summary": attendance_summary()}


def security() -> dict:
    return {
        "securityCategories": fetch_all("select event_type as name, count(*) as value from events group by event_type order by value desc limit 8"),
        "eventsByHour": fetch_all("select substr(timestamp,12,2) as hour, count(*) as events from events group by substr(timestamp,12,2) order by hour"),
        "bySeverity": fetch_all("select severity, count(*) as count from events group by severity"),
    }


def objects() -> dict:
    rows = fetch_all(
        """
        select event_type as class_name, count(*) as count
        from events
        where event_type like '%OBJECT%' or event_type like '%WEAPON%' or event_type like '%FIRE%' or event_type like '%SMOKE%'
        group by event_type order by count desc limit 20
        """
    )
    return {"objects": rows}

