from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

from ..database import fetch_all
from ..config import SCHOOL_DAYS, SCHOOL_HOLIDAYS, local_today
from ..database import execute


def overview(year: int | None = None, month: int | None = None) -> dict[str, Any]:
    """Expose academic metadata without requiring biometric records to contain it."""
    today = local_today()
    year = year or today.year
    month = month or today.month
    people = fetch_all("select id, name, role, metadata_json from people order by name")
    subjects: set[str] = set()
    profiles = []
    for person in people:
        person_subjects = metadata_subjects(person.get("metadata_json"))
        subjects.update(person_subjects)
        profiles.append({"person_id": person["id"], "name": person["name"], "subjects": person_subjects})
    start = date(year, month, 1)
    end = date(year, month, monthrange(year, month)[1])
    cutoff = min(local_today(), end)
    recorded = fetch_all("select person_id, date from attendance where date between ? and ? and clock_in is not null", [start.isoformat(), cutoff.isoformat()])
    recorded_keys = {(row["person_id"], row["date"]) for row in recorded}
    stored = fetch_all(
        "select a.*, p.name from absence_records a join people p on p.id=a.person_id where a.absence_date between ? and ?",
        [start.isoformat(), cutoff.isoformat()],
    )
    stored_keys = {(row["person_id"], row["absence_date"]) for row in stored}
    absences = [
        {
            "id": row["id"], "person_id": row["person_id"], "name": row["name"],
            "date": row["absence_date"], "subject": row.get("subject") or None,
            "status": row["status"].replace("_", " ").title(),
            "absence_type": row["status"], "reason": row.get("reason"),
            "official": bool(row.get("is_official")), "source": row.get("source"),
        }
        for row in stored
    ]
    day = start
    while day <= cutoff:
        if day.weekday() in SCHOOL_DAYS and day.isoformat() not in SCHOOL_HOLIDAYS:
            for person in people:
                if (person["id"], day.isoformat()) not in recorded_keys and (person["id"], day.isoformat()) not in stored_keys:
                    absences.append({"person_id": person["id"], "name": person["name"], "date": day.isoformat(), "status": "Inferred absence", "absence_type": "inferred", "official": False, "source": "system"})
        day += timedelta(days=1)
    absences.sort(key=lambda row: (row["date"], row["name"]), reverse=True)
    return {
        "year": year,
        "month": month,
        "subjects": sorted(subjects),
        "profiles": profiles,
        "absence_records": absences,
        "school_policy": {
            "school_days": list(SCHOOL_DAYS),
            "holidays": list(SCHOOL_HOLIDAYS),
            "absence_is_inferred_until_confirmed": True,
        },
        "absence_note": "Absence is inferred from a missing attendance record. Confirm policy before treating it as an official absence.",
    }


def metadata_subjects(value: Any) -> list[str]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return []
    subjects = parsed.get("subjects", []) if isinstance(parsed, dict) else []
    if isinstance(subjects, str):
        subjects = [subjects]
    return [str(subject) for subject in subjects if str(subject).strip()]


def list_schedules(active_only: bool = False) -> list[dict[str, Any]]:
    where = " where active=1" if active_only else ""
    return fetch_all(f"select * from attendance_schedules{where} order by class_name, weekday, start_time", [])


def create_schedule(
    class_name: str,
    subject: str | None,
    weekday: int,
    start_time: str,
    end_time: str | None,
    grace_minutes: int,
) -> dict[str, Any]:
    if not 0 <= int(weekday) <= 6:
        raise ValueError("weekday must be between 0 and 6")
    if not str(start_time).strip() or len(str(start_time).strip()) > 5:
        raise ValueError("start_time must use HH:MM")
    class_name = str(class_name).strip()[:120]
    if not class_name:
        raise ValueError("class_name is required")
    subject_value = (subject or "").strip()[:120]
    schedule_id = execute(
        """
        insert into attendance_schedules (class_name, subject, weekday, start_time, end_time, grace_minutes)
        values (?, ?, ?, ?, ?, ?)
        on conflict(class_name, subject, weekday, start_time) do update set
          end_time=excluded.end_time, grace_minutes=excluded.grace_minutes, active=1
        """,
        [class_name, subject_value, int(weekday), str(start_time).strip(), (end_time or "").strip()[:5] or None, max(0, min(int(grace_minutes), 240))],
    )
    row = fetch_all("select * from attendance_schedules where class_name=? and weekday=? and start_time=? order by id desc limit 1", [class_name, int(weekday), str(start_time).strip()])[0]
    return row


def deactivate_schedule(schedule_id: int) -> dict[str, Any]:
    row = fetch_all("select * from attendance_schedules where id=?", [schedule_id])
    if not row:
        raise ValueError("Schedule was not found")
    execute("update attendance_schedules set active=0 where id=?", [schedule_id])
    return fetch_all("select * from attendance_schedules where id=?", [schedule_id])[0]
