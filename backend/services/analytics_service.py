from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta

from ..config import ORGANIZATION_ID, SITE_ID, local_today
from ..database import fetch_all, fetch_one
from .attendance_service import attendance_summary
from .event_service import event_category
from .runtime_service import live_state


def overview() -> dict:
    live = live_state()
    people = fetch_one("select count(*) as c from people where organization_id=? and site_id=?", [ORGANIZATION_ID, SITE_ID]) or {"c": 0}
    alerts = fetch_one("select count(*) as c from alert_log where status='sent' and organization_id=? and site_id=?", [ORGANIZATION_ID, SITE_ID]) or {"c": 0}
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
    days = max(1, min(int(days), 366))
    end = local_today()
    start = end - timedelta(days=days - 1)
    by_day = fetch_all(
        """
        select date as day,
               count(*) as present,
               sum(case when late_minutes > 0 then 1 else 0 end) as late
        from attendance
        where date between ? and ? and organization_id=? and site_id=?
        group by date order by date desc
        """,
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    status_rows = fetch_all(
        """
        select case when clock_out is not null then 'Left'
                    when late_minutes > 0 then 'Late'
                    when clock_in is not null then 'Present'
                    else 'Not Yet Detected' end as name, count(*) as value
        from attendance where date between ? and ? and organization_id=? and site_id=? group by name order by value desc
        """
        , [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID]
    )
    methods = fetch_all(
        "select case when lower(coalesce(notes,'')) like '%center%' or lower(coalesce(notes,'')) like '%verified%' then 'Center Verified' when lower(coalesce(notes,'')) like '%manual%' then 'Manual' else 'Automatic' end as name, count(*) as value from attendance where date between ? and ? and organization_id=? and site_id=? group by name",
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    roster = fetch_one("select count(*) as registered, sum(case when lower(coalesce(metadata_json,'')) not like '%\"active\": false%' then 1 else 0 end) as active from people where organization_id=? and site_id=?", [ORGANIZATION_ID, SITE_ID]) or {}
    seen = fetch_one("select count(distinct person_id) as seen from attendance where date=? and organization_id=? and site_id=?", [local_today().isoformat(), ORGANIZATION_ID, SITE_ID]) or {}
    event_total = fetch_one("select count(*) as total from events where date(timestamp) between ? and ? and organization_id=? and site_id=?", [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID]) or {}
    absence_rows = fetch_all(
        "select status as name, count(*) as value from absence_records where absence_date between ? and ? and organization_id=? and site_id=? group by status order by value desc",
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    class_people = fetch_all("select id, metadata_json from people where organization_id=? and site_id=?", [ORGANIZATION_ID, SITE_ID])
    class_attendance = fetch_all(
        "select distinct person_id from attendance where date between ? and ? and clock_in is not null and organization_id=? and site_id=?",
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    present_ids = {row["person_id"] for row in class_attendance}
    class_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"present": 0, "roster": 0})
    for person in class_people:
        try:
            metadata = json.loads(person.get("metadata_json") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        class_name = metadata.get("class") or metadata.get("class_name") or "Unassigned" if isinstance(metadata, dict) else "Unassigned"
        totals = class_totals[str(class_name)]
        totals["roster"] += 1
        if person["id"] in present_ids:
            totals["present"] += 1
    class_rows = [{"name": name, **totals} for name, totals in sorted(class_totals.items())]
    evidence_rows = fetch_all(
        "select decision as name, count(*) as value from recognition_evidence where date(observed_at) between ? and ? and organization_id=? and site_id=? group by decision order by value desc",
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    incidents = fetch_one("select count(*) as total, sum(case when status not in ('dismissed','resolved') then 1 else 0 end) as open from incidents where organization_id=? and site_id=?", [ORGANIZATION_ID, SITE_ID]) or {}
    return {
        "attendanceByDay": list(reversed(by_day)),
        "summary": attendance_summary(),
        "byStatus": status_rows,
        "methodSplit": methods,
        "rosterTotals": {"registered": roster.get("registered", 0) or 0, "active": roster.get("active", 0) or 0, "seenToday": seen.get("seen", 0) or 0},
        "totalEvents": event_total.get("total", 0) or 0,
        "absenceSplit": absence_rows,
        "classCoverage": class_rows,
        "recognitionDecisions": evidence_rows,
        "incidentTotals": {"total": incidents.get("total", 0) or 0, "open": incidents.get("open", 0) or 0},
    }


def security(days: int = 30) -> dict:
    days = max(1, min(int(days), 366))
    end = local_today()
    start = end - timedelta(days=days - 1)
    rows = fetch_all(
        "select event_type, severity, timestamp from events where date(timestamp) between ? and ? and organization_id=? and site_id=?",
        [start.isoformat(), end.isoformat(), ORGANIZATION_ID, SITE_ID],
    )
    security_rows = [
        row for row in rows
        if event_category(str(row.get("event_type") or "")) in {"Security", "Safety", "Object"}
        or "SPOOF" in str(row.get("event_type") or "").upper()
    ]
    category_totals: dict[str, int] = defaultdict(int)
    hour_totals: dict[str, int] = defaultdict(int)
    severity_totals: dict[int, int] = defaultdict(int)
    for row in security_rows:
        category_totals[str(row.get("event_type") or "UNKNOWN")] += 1
        timestamp = str(row.get("timestamp") or "")
        hour_totals[timestamp[11:13] if len(timestamp) >= 13 else "--"] += 1
        try:
            severity_totals[int(row.get("severity") or 0)] += 1
        except (TypeError, ValueError):
            severity_totals[0] += 1
    return {
        "securityObservationTotal": len(security_rows),
        "securityCategories": [
            {"name": name, "value": value}
            for name, value in sorted(category_totals.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
        "eventsByHour": [
            {"hour": hour, "events": hour_totals[hour]}
            for hour in sorted(hour_totals)
        ],
        "bySeverity": [
            {"severity": severity, "count": count}
            for severity, count in sorted(severity_totals.items())
        ],
    }


def objects() -> dict:
    rows = fetch_all(
        """
        select event_type as class_name, count(*) as count
        from events
        where (event_type like '%OBJECT%' or event_type like '%WEAPON%' or event_type like '%FIRE%' or event_type like '%SMOKE%')
          and organization_id=? and site_id=?
        group by event_type order by count desc limit 20
        """
        , [ORGANIZATION_ID, SITE_ID]
    )
    return {"objects": rows}
