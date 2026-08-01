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
    status_rows = fetch_all(
        """
        select case when clock_out is not null then 'Left'
                    when late_minutes > 0 then 'Late'
                    when clock_in is not null then 'Present'
                    else 'Not Yet Detected' end as name, count(*) as value
        from attendance group by name order by value desc
        """
    )
    methods = fetch_all(
        "select case when lower(coalesce(notes,'')) like '%manual%' then 'Manual' else 'Automatic' end as name, count(*) as value from attendance group by name"
    )
    roster = fetch_one("select count(*) as registered, sum(case when lower(coalesce(metadata_json,'')) not like '%\"active\": false%' then 1 else 0 end) as active from people") or {}
    seen = fetch_one("select count(distinct person_id) as seen from attendance where date=date('now','localtime')") or {}
    event_total = fetch_one("select count(*) as total from events") or {}
    return {
        "attendanceByDay": list(reversed(by_day)),
        "summary": attendance_summary(),
        "byStatus": status_rows,
        "methodSplit": methods,
        "rosterTotals": {"registered": roster.get("registered", 0) or 0, "active": roster.get("active", 0) or 0, "seenToday": seen.get("seen", 0) or 0},
        "totalEvents": event_total.get("total", 0) or 0,
    }


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
