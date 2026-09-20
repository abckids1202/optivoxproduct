from __future__ import annotations

import csv
import io
import json
from calendar import monthrange
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..database import execute, fetch_all, fetch_one, transaction
from ..config import DEVICE_ID, ORGANIZATION_ID, SITE_ID, local_now, local_today
from .audit_service import record_action_in_connection
from .outbox_service import enqueue_in_connection


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
        "attendanceStatus": row.get("attendance_status") or "recorded",
        "absenceType": row.get("absence_type"),
        "official": bool(row.get("is_official", 1)),
        "subject": row.get("subject"),
        "scheduleKey": row.get("schedule_key"),
        "expectedStart": row.get("expected_start"),
        "expectedEnd": row.get("expected_end"),
        "earlyDepartureMinutes": row.get("early_departure_minutes") or 0,
        "confidence": row.get("recognition_confidence"),
        "presenceSessionId": row.get("presence_session_id"),
        "recognitionEvidenceId": row.get("recognition_evidence_id"),
        "decisionSource": row.get("decision_source") or attendance_method(row.get("notes")),
        "identityState": row.get("identity_state"),
        "livenessStatus": row.get("liveness_status"),
        "policyVersion": row.get("policy_version"),
        "sourceFrameId": row.get("source_frame_id"),
        # The evidence file is served through an authorized event endpoint;
        # local filesystem paths must not cross the API boundary.
        "evidenceAvailable": bool(row.get("evidence_path")),
        "evidencePath": None,
        "method": attendance_method(row.get("notes")),
        "camera": row.get("camera_id"),
        "location": row.get("location"),
        "lastSeen": row.get("last_seen_at") or row.get("last_seen") or row.get("clock_in"),
        "clockOutSource": row.get("clock_out_source"),
        "active": True,
    }


def metadata_class(value: Any) -> str | None:
    import json
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("class", "class_name", "grade", "section"):
        candidate = parsed.get(key)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def attendance_method(value: Any) -> str:
    text = str(value or "").lower()
    if "center" in text or "verified" in text:
        return "Center Verified"
    if "manual" in text:
        return "Manual"
    return "Automatic"


def display_time(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return text[11:16] if len(text) >= 16 and "T" in text else text


def list_attendance(today_only: bool = False, limit: int = 200, offset: int = 0, person_id: int | None = None) -> list[dict[str, Any]]:
    where = ["a.organization_id = ?", "a.site_id = ?", "p.organization_id = ?", "p.site_id = ?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID, ORGANIZATION_ID, SITE_ID]
    if today_only:
        where.append("a.date = ?")
        params.append(local_today().isoformat())
    if person_id:
        where.append("a.person_id = ?")
        params.append(person_id)
    sql = """
      select a.*, p.name, p.role, p.metadata_json,
             (select max(e.timestamp) from events e where e.person_id = p.id and e.organization_id=? and e.site_id=?) as last_seen,
             (select e.confidence from events e where e.person_id = p.id and e.organization_id=? and e.site_id=? order by e.timestamp desc limit 1) as recognition_confidence
      from attendance a
      join people p on p.id = a.person_id
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by a.date desc, a.clock_in desc limit ? offset ?"
    params.extend([max(1, min(limit, 500)), max(0, offset)])
    params = [ORGANIZATION_ID, SITE_ID, ORGANIZATION_ID, SITE_ID] + params
    return [normalize_attendance(row) for row in fetch_all(sql, params)]


def today_attendance() -> list[dict[str, Any]]:
    return list_attendance(today_only=True)


def attendance_summary() -> dict[str, Any]:
    rows = today_attendance()
    roster = fetch_one("select count(*) as c from people where organization_id=? and site_id=? and lower(coalesce(metadata_json,'')) not like '%\"active\": false%'", [ORGANIZATION_ID, SITE_ID]) or {"c": 0}
    detected = len(rows)
    today_absences = fetch_one(
        "select sum(case when is_official=1 then 1 else 0 end) as official, sum(case when is_official=0 then 1 else 0 end) as inferred from absence_records where absence_date=? and organization_id=? and site_id=?",
        [local_today().isoformat(), ORGANIZATION_ID, SITE_ID],
    ) or {}
    return {
        "total": int(roster.get("c") or 0),
        "present": sum(1 for r in rows if r["status"] in ("Present", "Late")),
        "late": sum(1 for r in rows if r["status"] == "Late"),
        "left": sum(1 for r in rows if r["status"] == "Left"),
        "detected": detected,
        "not_yet_detected": max(0, int(roster.get("c") or 0) - detected),
        "official_absences": int(today_absences.get("official") or 0),
        "inferred_absences": int(today_absences.get("inferred") or 0),
    }


def person_attendance(person_id: int) -> list[dict[str, Any]]:
    return list_attendance(person_id=person_id)


def attendance_calendar(year: int | None = None, month: int | None = None) -> dict[str, Any]:
    """Return a complete roster matrix, including people with no attendance row."""
    today = local_today()
    year = year or today.year
    month = month or today.month
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    days_in_month = monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01"
    end = f"{year:04d}-{month:02d}-{days_in_month:02d}"
    people = fetch_all("select id, name, role, metadata_json from people where organization_id=? and site_id=? order by name", [ORGANIZATION_ID, SITE_ID])
    rows = fetch_all(
        """
        select a.*, p.name, p.role, p.metadata_json
        from attendance a join people p on p.id=a.person_id
        where a.date between ? and ? and a.organization_id=? and a.site_id=? and p.organization_id=? and p.site_id=?
        order by a.date, p.name
        """,
        [start, end, ORGANIZATION_ID, SITE_ID, ORGANIZATION_ID, SITE_ID],
    )
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_attendance(row)
        records[f"{row['person_id']}:{row['date']}"] = normalized
    absence_rows = fetch_all(
        """
        select a.*, p.name, p.role, p.metadata_json
        from absence_records a join people p on p.id=a.person_id
        where a.absence_date between ? and ? and a.organization_id=? and a.site_id=? and p.organization_id=? and p.site_id=?
        order by a.absence_date desc, p.name
        """,
        [start, end, ORGANIZATION_ID, SITE_ID, ORGANIZATION_ID, SITE_ID],
    )
    for absence in absence_rows:
        key = f"{absence['person_id']}:{absence['absence_date']}"
        if key in records:
            continue
        records[key] = {
            "id": None,
            "person_id": absence["person_id"],
            "name": absence.get("name"),
            "person": absence.get("name"),
            "role": absence.get("role"),
            "className": metadata_class(absence.get("metadata_json")),
            "date": absence["absence_date"],
            "status": absence["status"].replace("_", " ").title(),
            "absenceStatus": absence["status"],
            "absenceReason": absence.get("reason"),
            "official": bool(absence.get("is_official")),
            "subject": absence.get("subject") or None,
            "decisionSource": absence.get("source") or "system",
        }
    days = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]
    roster = [{
        "id": person["id"],
        "name": person["name"],
        "role": person.get("role"),
        "subjects": metadata_subjects(person.get("metadata_json")),
        "records": {day: records.get(f"{person['id']}:{day}") for day in days},
    } for person in people]
    return {
        "year": year,
        "month": month,
        "days": days,
        "people": roster,
        "absenceRecords": [
            {
                "id": row["id"],
                "person_id": row["person_id"],
                "date": row["absence_date"],
                "subject": row.get("subject") or None,
                "status": row["status"],
                "reason": row.get("reason"),
                "official": bool(row.get("is_official")),
                "source": row.get("source"),
            }
            for row in absence_rows
        ],
    }


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


def list_absences(person_id: int | None = None, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    where = ["a.organization_id=?", "a.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    if person_id is not None:
        where.append("a.person_id=?")
        params.append(person_id)
    if start:
        where.append("a.absence_date>=?")
        params.append(start)
    if end:
        where.append("a.absence_date<=?")
        params.append(end)
    sql = """
        select a.*, p.name, p.role, p.metadata_json
        from absence_records a join people p on p.id=a.person_id
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by a.absence_date desc, p.name"
    return [
        {
            "id": row["id"], "person_id": row["person_id"], "name": row["name"],
            "date": row["absence_date"], "subject": row.get("subject") or None,
            "status": row["status"], "reason": row.get("reason"),
            "official": bool(row.get("is_official")), "source": row.get("source"),
            "actor_id": row.get("actor_id"), "created_at": row.get("created_at"),
        }
        for row in fetch_all(sql, params)
    ]


def record_absence(
    person_id: int,
    absence_date: str,
    status: str,
    subject: str | None = None,
    reason: str | None = None,
    actor_id: str | None = None,
) -> dict[str, Any]:
    if not fetch_one("select id from people where id=? and organization_id=? and site_id=?", [person_id, ORGANIZATION_ID, SITE_ID]):
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    try:
        datetime.strptime(absence_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail={"code": "INVALID_DATE", "message": "Use an absence date in YYYY-MM-DD format."})
    status = str(status or "").strip().lower()
    if status not in {"inferred", "official", "excused"}:
        raise HTTPException(status_code=422, detail={"code": "INVALID_ABSENCE_STATUS", "message": "Use inferred, official, or excused."})
    reason = (reason or "").strip()[:500] or None
    official = 1 if status in {"official", "excused"} else 0
    subject_value = (subject or "").strip()[:120]
    with transaction(immediate=True) as con:
        con.execute(
            """
            insert into absence_records (person_id, absence_date, subject, status, reason, source, is_official, actor_id, organization_id, site_id, device_id)
            values (?, ?, ?, ?, ?, 'operator', ?, ?, ?, ?, ?)
            on conflict(person_id, absence_date, subject) do update set
              status=excluded.status, reason=excluded.reason, source=excluded.source,
              is_official=excluded.is_official, actor_id=excluded.actor_id, updated_at=datetime('now')
            """,
            [person_id, absence_date, subject_value, status, reason, official, actor_id, ORGANIZATION_ID, SITE_ID, DEVICE_ID],
        )
        record_action_in_connection(
            con, "attendance.absence_record", "absence", person_id,
            {"date": absence_date, "status": status, "subject": subject_value},
            actor_id=actor_id,
        )
        enqueue_in_connection(
            con,
            "attendance.absence_recorded",
            {"person_id": person_id, "date": absence_date, "status": status, "subject": subject_value},
            event_id=f"absence:{ORGANIZATION_ID}:{SITE_ID}:{person_id}:{absence_date}:{subject_value}",
            aggregate_type="absence",
            aggregate_id=person_id,
        )
        row = con.execute(
            "select a.*, p.name from absence_records a join people p on p.id=a.person_id where a.person_id=? and a.absence_date=? and a.subject=? and a.organization_id=? and a.site_id=?",
            [person_id, absence_date, subject_value, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        row = dict(row or {})
    return {
        "id": row["id"], "person_id": row["person_id"], "name": row["name"],
        "date": row["absence_date"], "subject": row.get("subject") or None,
        "status": row["status"], "reason": row.get("reason"), "official": bool(row.get("is_official")),
    }


def clock_in(person_id: int, actor_id: str | None = None) -> dict[str, Any]:
    today = local_today().isoformat()
    now = local_now().isoformat(timespec="seconds")
    with transaction(immediate=True) as con:
        person = con.execute(
            "select id, metadata_json from people where id=? and organization_id=? and site_id=?",
            [person_id, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        if not person:
            raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
        try:
            person_metadata = json.loads(person["metadata_json"] or "{}")
        except (TypeError, ValueError):
            person_metadata = {}
        if isinstance(person_metadata, dict) and not person_metadata.get("active", True):
            raise HTTPException(status_code=409, detail={"code": "PERSON_INACTIVE", "message": "Inactive profiles cannot be clocked in."})
        existing = con.execute(
            "select * from attendance where person_id=? and date=? and organization_id=? and site_id=?",
            [person_id, today, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        if existing and existing["clock_in"]:
            return {"status": "already_clocked_in", "record": normalize_attendance({**dict(existing), "name": None, "role": None, "metadata_json": None})}
        con.execute(
            """
            insert into attendance (person_id, date, clock_in, late_minutes, notes, attendance_status, decision_source, is_official, organization_id, site_id, device_id)
            values (?, ?, ?, 0, 'manual_web', 'manual', 'manual', 1, ?, ?, ?)
            on conflict(person_id, date) do update set clock_in=coalesce(attendance.clock_in, excluded.clock_in), notes='manual_web', decision_source='manual', organization_id=excluded.organization_id, site_id=excluded.site_id, device_id=excluded.device_id
            """,
            [person_id, today, now, ORGANIZATION_ID, SITE_ID, DEVICE_ID],
        )
        record_action_in_connection(
            con, "attendance.clock_in", "attendance", person_id,
            {"method": "Manual", "date": today}, actor_id=actor_id,
        )
        attendance_row = con.execute(
            "select id from attendance where person_id=? and date=? and organization_id=? and site_id=?",
            [person_id, today, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        enqueue_in_connection(
            con,
            "attendance.clocked_in",
            {"person_id": person_id, "attendance_id": attendance_row["id"] if attendance_row else None, "date": today, "method": "manual"},
            event_id=f"attendance-in:{ORGANIZATION_ID}:{SITE_ID}:{person_id}:{today}",
            aggregate_type="attendance",
            aggregate_id=attendance_row["id"] if attendance_row else person_id,
        )
    return {"status": "clocked_in", "person_id": person_id}


def clock_out(person_id: int, actor_id: str | None = None) -> dict[str, Any]:
    today = local_today().isoformat()
    now = local_now().isoformat(timespec="seconds")
    with transaction(immediate=True) as con:
        person = con.execute(
            "select id from people where id=? and organization_id=? and site_id=?",
            [person_id, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        if not person:
            raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
        existing = con.execute(
            "select * from attendance where person_id=? and date=? and organization_id=? and site_id=?",
            [person_id, today, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        if not existing or not existing["clock_in"]:
            raise HTTPException(status_code=400, detail={"code": "NOT_CLOCKED_IN", "message": "Person is not clocked in today."})
        con.execute(
            "update attendance set clock_out=?, notes='manual_web', decision_source='manual' where person_id=? and date=? and organization_id=? and site_id=?",
            [now, person_id, today, ORGANIZATION_ID, SITE_ID],
        )
        record_action_in_connection(
            con, "attendance.clock_out", "attendance", person_id,
            {"method": "Manual", "date": today}, actor_id=actor_id,
        )
        enqueue_in_connection(
            con,
            "attendance.clocked_out",
            {"person_id": person_id, "attendance_id": existing["id"], "date": today, "method": "manual"},
            event_id=f"attendance-out:{ORGANIZATION_ID}:{SITE_ID}:{person_id}:{today}",
            aggregate_type="attendance",
            aggregate_id=existing["id"],
        )
    return {"status": "clocked_out", "person_id": person_id}


def correct_attendance(
    person_id: int,
    attendance_date: str,
    clock_in: str | None,
    clock_out: str | None,
    late_minutes: int,
    reason: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    person = fetch_one("select id from people where id=? and organization_id=? and site_id=?", [person_id, ORGANIZATION_ID, SITE_ID])
    if not person:
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    try:
        datetime.strptime(attendance_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail={"code": "INVALID_DATE", "message": "Use an attendance date in YYYY-MM-DD format."})
    reason = (reason or "").strip()
    if len(reason) < 3:
        raise HTTPException(status_code=422, detail={"code": "CORRECTION_REASON_REQUIRED", "message": "A correction reason is required."})
    work_minutes = 0
    if clock_in and clock_out:
        try:
            start = datetime.fromisoformat(clock_in.replace("Z", "+00:00"))
            end = datetime.fromisoformat(clock_out.replace("Z", "+00:00"))
            work_minutes = max(0, int((end - start).total_seconds() // 60))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail={"code": "INVALID_TIME", "message": "Clock times must be ISO timestamps."})
    notes = f"corrected: {reason[:350]}"
    with transaction(immediate=True) as con:
        existing = con.execute(
            "select * from attendance where person_id=? and date=? and organization_id=? and site_id=?",
            [person_id, attendance_date, ORGANIZATION_ID, SITE_ID],
        ).fetchone()
        before = dict(existing or {"person_id": person_id, "date": attendance_date})
        if existing:
            con.execute(
                "update attendance set clock_in=?, clock_out=?, work_minutes=?, late_minutes=?, notes=?, attendance_status='corrected', decision_source='manual_correction', is_official=1 where person_id=? and date=? and organization_id=? and site_id=?",
                [clock_in, clock_out, work_minutes, max(0, min(int(late_minutes), 1440)), notes, person_id, attendance_date, ORGANIZATION_ID, SITE_ID],
            )
            attendance_id = existing["id"]
        else:
            cur = con.execute(
                "insert into attendance (person_id, date, clock_in, clock_out, work_minutes, late_minutes, notes, attendance_status, decision_source, is_official, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?, ?, 'corrected', 'manual_correction', 1, ?, ?, ?)",
                [person_id, attendance_date, clock_in, clock_out, work_minutes, max(0, min(int(late_minutes), 1440)), notes, ORGANIZATION_ID, SITE_ID, DEVICE_ID],
            )
            attendance_id = int(cur.lastrowid)
        after = dict(con.execute("select * from attendance where id=?", [attendance_id]).fetchone() or {})
        con.execute(
            "insert into attendance_corrections (attendance_id, person_id, attendance_date, before_json, after_json, reason, actor_id, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [attendance_id, person_id, attendance_date, json.dumps(before, default=str), json.dumps(after, default=str), reason, actor_id, ORGANIZATION_ID, SITE_ID, DEVICE_ID],
        )
        record_action_in_connection(
            con, "attendance.correct", "attendance", attendance_id,
            {"person_id": person_id, "date": attendance_date, "reason": reason},
            actor_id=actor_id,
        )
        enqueue_in_connection(
            con,
            "attendance.corrected",
            {"person_id": person_id, "attendance_id": attendance_id, "date": attendance_date, "reason": reason},
            event_id=f"attendance-correction:{attendance_id}:{attendance_date}",
            aggregate_type="attendance",
            aggregate_id=attendance_id,
        )
    return normalize_attendance({**after, "name": None, "role": None, "metadata_json": None})


def export_csv() -> StreamingResponse:
    rows = list_attendance(limit=1000)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["name", "date", "status", "clock_in", "clock_out", "late_minutes", "camera", "location"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in writer.fieldnames})
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=optivox_attendance.csv"})
