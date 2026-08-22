from __future__ import annotations

from typing import Any
import json

from fastapi import HTTPException

from ..database import execute, fetch_all, fetch_one
from ..config import local_today
from .attendance_service import attendance_status
from .audit_service import record_action


def normalize_person(row: dict[str, Any]) -> dict[str, Any]:
    metadata = parse_metadata(row.get("metadata_json"))
    active = True
    active = bool(metadata.get("active", True))
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row.get("role"),
        "className": metadata.get("class") or metadata.get("class_name"),
        "subjects": normalized_subjects(metadata.get("subjects", [])),
        "active": active,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "thumbnail": row.get("thumbnail_path"),
        "samples": row.get("sample_count") or 1,
        "lastSeen": row.get("last_seen"),
        "status": row.get("attendance_status") or "Not Yet Detected",
        "attendance_today": row.get("attendance_status"),
      }


def list_people() -> list[dict[str, Any]]:
    today = local_today().isoformat()
    rows = fetch_all(
        """
        select p.*,
               (select max(e.timestamp) from events e where e.person_id=p.id) as last_seen,
               (select a.clock_in from attendance a where a.person_id=p.id and a.date=?) as today_clock_in,
               (select a.clock_out from attendance a where a.person_id=p.id and a.date=?) as today_clock_out,
               (select a.late_minutes from attendance a where a.person_id=p.id and a.date=?) as today_late
        from people p order by p.name
        """,
        [today, today, today],
    )
    out = []
    for row in rows:
        status = "Not Yet Detected"
        if row.get("today_clock_out"):
            status = "Left"
        elif row.get("today_clock_in"):
            status = "Late" if int(row.get("today_late") or 0) > 0 else "Present"
        out.append(normalize_person({**row, "attendance_status": status}))
    return out


def get_person(person_id: int) -> dict[str, Any]:
    row = fetch_one("select * from people where id=?", [person_id])
    if not row:
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    person = normalize_person(row)
    person["recent_events"] = fetch_all("select * from events where person_id=? order by timestamp desc limit 20", [person_id])
    person["attendance_summary"] = fetch_all("select * from attendance where person_id=? order by date desc limit 30", [person_id])
    return person


def update_person(person_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    current = fetch_one("select * from people where id=?", [person_id])
    if not current:
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    allowed = {k: v for k, v in payload.items() if k in {"name", "role"} and v is not None}
    if "name" in allowed:
        allowed["name"] = str(allowed["name"]).strip()
        if not allowed["name"]:
            raise HTTPException(status_code=422, detail={"code": "INVALID_NAME", "message": "A person name is required."})
        duplicate = fetch_one("select id from people where lower(name)=lower(?) and id<>?", [allowed["name"], person_id])
        if duplicate:
            raise HTTPException(status_code=409, detail={"code": "DUPLICATE_PERSON", "message": "Another person already has this name."})
    if not allowed:
        return get_person(person_id)
    sets = ", ".join(f"{k}=?" for k in allowed)
    params = list(allowed.values()) + [person_id]
    execute(f"update people set {sets}, updated_at=datetime('now') where id=?", params)
    record_action("people.update", "person", person_id, {"fields": sorted(allowed)})
    return get_person(person_id)


def parse_metadata(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def normalized_subjects(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def set_enabled(person_id: int, enabled: bool) -> dict[str, Any]:
    if not fetch_one("select id from people where id=?", [person_id]):
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    metadata = parse_metadata(fetch_one("select metadata_json from people where id=?", [person_id]).get("metadata_json"))
    metadata["active"] = enabled
    execute("update people set metadata_json=?, updated_at=datetime('now') where id=?", [json.dumps(metadata), person_id])
    record_action("people.enable" if enabled else "people.disable", "person", person_id, {"active": enabled})
    return get_person(person_id)
