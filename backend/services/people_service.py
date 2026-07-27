from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..database import execute, fetch_all, fetch_one
from .attendance_service import attendance_status


def normalize_person(row: dict[str, Any]) -> dict[str, Any]:
    active = True
    if row.get("metadata_json") and '"active": false' in str(row.get("metadata_json")).lower():
        active = False
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row.get("role"),
        "className": None,
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
    rows = fetch_all(
        """
        select p.*,
               (select max(e.timestamp) from events e where e.person_id=p.id) as last_seen,
               (select a.clock_in from attendance a where a.person_id=p.id and a.date=date('now','localtime')) as today_clock_in,
               (select a.clock_out from attendance a where a.person_id=p.id and a.date=date('now','localtime')) as today_clock_out,
               (select a.late_minutes from attendance a where a.person_id=p.id and a.date=date('now','localtime')) as today_late
        from people p order by p.name
        """
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
    allowed = {k: v for k, v in payload.items() if k in {"name", "role"} and v is not None}
    if not allowed:
        return get_person(person_id)
    sets = ", ".join(f"{k}=?" for k in allowed)
    params = list(allowed.values()) + [person_id]
    execute(f"update people set {sets}, updated_at=datetime('now') where id=?", params)
    return get_person(person_id)


def set_enabled(person_id: int, enabled: bool) -> dict[str, Any]:
    if not fetch_one("select id from people where id=?", [person_id]):
        raise HTTPException(status_code=404, detail={"code": "PERSON_NOT_FOUND", "message": "Person was not found."})
    execute("update people set metadata_json=?, updated_at=datetime('now') where id=?", [f'{{"active": {str(enabled).lower()}}}', person_id])
    return get_person(person_id)

