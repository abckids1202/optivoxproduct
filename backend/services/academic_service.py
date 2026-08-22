from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

from ..database import fetch_all
from ..config import local_today


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
    absences = []
    day = start
    while day <= cutoff:
        if day.weekday() < 5:
            for person in people:
                if (person["id"], day.isoformat()) not in recorded_keys:
                    absences.append({"person_id": person["id"], "name": person["name"], "date": day.isoformat(), "status": "Inferred absence"})
        day += timedelta(days=1)
    absences.sort(key=lambda row: (row["date"], row["name"]), reverse=True)
    return {
        "year": year,
        "month": month,
        "subjects": sorted(subjects),
        "profiles": profiles,
        "absence_records": absences,
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
