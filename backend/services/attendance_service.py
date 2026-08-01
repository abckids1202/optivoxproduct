from __future__ import annotations

import csv
import io
from calendar import monthrange
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..database import execute, fetch_all, fetch_one


def attendance_status(row: dict[str, Any]) -> str:
    if row.get("clock_out"):
        return "Left"
    if row.get("clock_in"):
        return "Late" if int(row.get("late_minutes") or 0) > 0 else "Present"
    return "Not Yet Detected"


def normalize_attendance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "person_id": row.get("person_id"),
        "name": row.get("name"),
        "person": row.get("name"),
        "role": row.get("role"),
        "className": metadata_class(row.get("metadata_json")),
        "date": row.get("date"),
        "status": attendance_status(row),
        "clockIn": display_time(row.get("clock_in")),
        "clockOut": display_time(row.get("clock_out")),
        "clock_in": row.get("clock_in"),
        "clock_out": row.get("clock_out"),
        "duration": row.get("work_minutes"),
        "work_minutes": row.get("work_minutes") or 0,
        "late": int(row.get("late_minutes") or 0) > 0,
        "late_minutes": row.get("late_minutes") or 0,
        "confidence": row.get("recognition_confidence"),
        "method": "Manual" if row.get("notes") and "manual" in str(row.get("notes")).lower() else "Automatic",
        "camera": row.get("camera_id"),
        "location": row.get("location"),
        "lastSeen": row.get("last_seen") or row.get("clock_in"),
        "active": True,
    }


def metadata_class(value: Any) -> str | None:
    return None


def display_time(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return text[11:16] if len(text) >= 16 and "T" in text else text


def list_attendance(today_only: bool = False, limit: int = 200, offset: int = 0, person_id: int | None = None) -> list[dict[str, Any]]:
    where = []
    params: list[Any] = []
    if today_only:
        where.append("a.date = date('now', 'localtime')")
    if person_id:
        where.append("a.person_id = ?")
        params.append(person_id)
    sql = """
      select a.*, p.name, p.role, p.metadata_json,
             (select max(e.timestamp) from events e where e.person_id = p.id) as last_seen,
             (select e.confidence from events e where e.person_id = p.id order by e.timestamp desc limit 1) as recognition_confidence
      from attendance a
      join people p on p.id = a.person_id
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by a.date desc, a.clock_in desc limit ? offset ?"
    params.extend([max(1, min(limit, 500)), max(0, offset)])
    return [normalize_attendance(row) for row in fetch_all(sql, params)]


def today_attendance() -> list[dict[str, Any]]:
    return list_attendance(today_only=True)


def attendance_summary() -> dict[str, Any]:
    rows = today_attendance()
    return {
        "total": len(rows),
        "present": sum(1 for r in rows if r["status"] in ("Present", "Late")),
        "late": sum(1 for r in rows if r["status"] == "Late"),
        "left": sum(1 for r in rows if r["status"] == "Left"),
        "not_yet_detected": 0,
    }


def person_attendance(person_id: int) -> list[dict[str, Any]]:
    return list_attendance(person_id=person_id)


def attendance_calendar(year: int | None = None, month: int | None = None) -> dict[str, Any]:
    """Return a complete roster matrix, including people with no attendance row."""
    today = date.today()
    year = year or today.year
    month = month or today.month
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    days_in_month = monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01"
    end = f"{year:04d}-{month:02d}-{days_in_month:02d}"
    people = fetch_all("select id, name, role, metadata_json from people order by name")
    rows = fetch_all(
        """
        select a.*, p.name, p.role, p.metadata_json
        from attendance a join people p on p.id=a.person_id
        where a.date between ? and ? order by a.date, p.name
        """,
        [start, end],
    )
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_attendance(row)
        records[f"{row['person_id']}:{row['date']}"] = normalized
    days = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]
    roster = [{
        "id": person["id"],
        "name": person["name"],
        "role": person.get("role"),
        "subjects": metadata_subjects(person.get("metadata_json")),
        "records": {day: records.get(f"{person['id']}:{day}") for day in days},
    } for person in people]
    return {"year": year, "month": month, "days": days, "people": roster}


def metadata_subjects(value: Any) -> list[str]:
    import json
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return []
    subjects = parsed.get("subjects", []) if isinstance(parsed, dict) else []
    if isinstance(subjects, str):
        subjects = [subjects]
    return [str(subject) for subject in subjects if str(subject).strip()]


def clock_in(person_id: int) -> dict[str, Any]:
    person = fetch_one("select id from people where id=?", [person_id])
    if not person:
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    today = date.today().isoformat()
    existing = fetch_one("select * from attendance where person_id=? and date=?", [person_id, today])
    if existing and existing.get("clock_in"):
        return {"status": "already_clocked_in", "record": normalize_attendance({**existing, "name": None, "role": None, "metadata_json": None})}
    execute(
        """
        insert into attendance (person_id, date, clock_in, late_minutes, notes)
        values (?, ?, datetime('now'), 0, 'manual_web')
        on conflict(person_id, date) do update set clock_in=coalesce(attendance.clock_in, excluded.clock_in), notes='manual_web'
        """,
        [person_id, today],
    )
    return {"status": "clocked_in", "person_id": person_id}


def clock_out(person_id: int) -> dict[str, Any]:
    person = fetch_one("select id from people where id=?", [person_id])
    if not person:
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    today = date.today().isoformat()
    existing = fetch_one("select * from attendance where person_id=? and date=?", [person_id, today])
    if not existing or not existing.get("clock_in"):
        raise HTTPException(status_code=400, detail={"code": "NOT_CLOCKED_IN", "message": "Person is not clocked in today."})
    execute("update attendance set clock_out=datetime('now'), notes='manual_web' where person_id=? and date=?", [person_id, today])
    return {"status": "clocked_out", "person_id": person_id}


def export_csv() -> StreamingResponse:
    rows = list_attendance(limit=1000)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["name", "date", "status", "clock_in", "clock_out", "late_minutes", "camera", "location"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in writer.fieldnames})
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=optivox_attendance.csv"})
