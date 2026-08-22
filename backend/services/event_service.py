from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..config import SNAPSHOTS_DIR
from ..database import fetch_all, fetch_one
from ..database import execute
from .audit_service import record_action


def severity_label(value: Any) -> str:
    try:
        value = int(value or 0)
    except (TypeError, ValueError):
        value = 0
    if value >= 3:
        return "Critical"
    if value >= 1:
        return "Warning"
    return "Normal"


def event_category(event_type: str) -> str:
    text = event_type.upper()
    if any(token in text for token in ["ATTENDANCE", "RECOGNITION", "SPOOF", "UNKNOWN"]):
        return "Identity"
    if any(token in text for token in ["FALL", "HANDS", "CROWD", "CONGESTION", "EVACUATION"]):
        return "Safety"
    if any(token in text for token in ["OBJECT", "WEAPON", "FIRE", "SMOKE"]):
        return "Object"
    return "System"


def parse_details(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return {"message": str(value)}


def normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    snapshot_path = row.get("snapshot_path")
    return {
        "id": row["id"],
        "time": row.get("timestamp"),
        "timestamp": row.get("timestamp"),
        "type": row.get("event_type"),
        "event_type": row.get("event_type"),
        "group": event_category(row.get("event_type", "")),
        "category": event_category(row.get("event_type", "")),
        "severity": severity_label(row.get("severity")),
        "person_id": row.get("person_id"),
        "person": row.get("person_name") or "System",
        "person_name": row.get("person_name"),
        "location": row.get("location"),
        "camera": row.get("camera_id"),
        "confidence": row.get("confidence") or 0,
        "details": parse_details(row.get("details_json")),
        "message": details_message(row.get("details_json")),
        "snapshot_available": bool(snapshot_path),
        "snapshot_url": f"/api/events/{row['id']}/snapshot" if snapshot_path else None,
        "review_status": row.get("review_status") or ("reviewed" if row.get("reviewed_at") else "open"),
        "review_note": row.get("review_note"),
        "reviewed": (row.get("review_status") or ("reviewed" if row.get("reviewed_at") else "open")) != "open",
    }


def details_message(value: Any) -> str:
    parsed = parse_details(value)
    if isinstance(parsed, dict):
        return str(parsed.get("message") or parsed)
    return str(parsed or "")


def list_events(limit: int = 100, offset: int = 0, event_type: str | None = None, severity: str | None = None) -> list[dict[str, Any]]:
    where = []
    params: list[Any] = []
    if event_type:
        where.append("e.event_type = ?")
        params.append(event_type)
    if severity:
        if severity.lower() == "critical":
            where.append("e.severity >= 3")
        elif severity.lower() in ("warning", "attention"):
            where.append("e.severity between 1 and 2")
        elif severity.lower() in ("normal", "info"):
            where.append("coalesce(e.severity, 0) = 0")
    sql = """
        select e.*, p.name as person_name
        from events e
        left join people p on p.id = e.person_id
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by e.timestamp desc limit ? offset ?"
    params.extend([max(1, min(limit, 500)), max(0, offset)])
    return [normalize_event(row) for row in fetch_all(sql, params)]


def get_event(event_id: int) -> dict[str, Any]:
    row = fetch_one(
        """
        select e.*, p.name as person_name
        from events e left join people p on p.id = e.person_id
        where e.id = ?
        """,
        [event_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    return normalize_event(row)


def event_summary() -> dict[str, Any]:
    rows = fetch_all("select event_type, severity, count(*) as count from events group by event_type, severity")
    return {
        "total": sum(r["count"] for r in rows),
        "by_type": rows,
        "critical": sum(r["count"] for r in rows if int(r["severity"] or 0) >= 3),
        "warning": sum(r["count"] for r in rows if 1 <= int(r["severity"] or 0) < 3),
    }


def review_event(event_id: int, action: str, note: str | None = None) -> dict[str, Any]:
    actions = {"confirm": "confirmed", "dismiss": "dismissed", "escalate": "escalated", "resolve": "resolved"}
    status = actions.get(action)
    if not status:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REVIEW_ACTION", "message": "Use confirm, dismiss, escalate, or resolve."})
    if not fetch_one("select id from events where id=?", [event_id]):
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    execute(
        "update events set review_status=?, review_note=?, reviewed_at=datetime('now'), reviewed_by='operator' where id=?",
        [status, (note or "").strip()[:500] or None, event_id],
    )
    record_action("event.review", "event", event_id, {"status": status, "note": note or ""})
    return get_event(event_id)


def snapshot_response(event_id: int) -> FileResponse:
    row = fetch_one("select snapshot_path from events where id=?", [event_id])
    if not row:
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    raw = row.get("snapshot_path")
    if not raw:
        raise HTTPException(status_code=404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "This event has no snapshot."})
    path = Path(raw)
    if not path.is_absolute():
        path = SNAPSHOTS_DIR / path
    path = path.resolve()
    if SNAPSHOTS_DIR not in path.parents and path.parent != SNAPSHOTS_DIR:
        raise HTTPException(status_code=400, detail={"code": "INVALID_SNAPSHOT_PATH", "message": "Snapshot path is outside the allowed directory."})
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "Snapshot file is missing."})
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
