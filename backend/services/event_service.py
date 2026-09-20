from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response

from ..config import ORGANIZATION_ID, SITE_ID, SNAPSHOTS_DIR
from ..database import fetch_all, fetch_one, transaction
from .audit_service import record_action, record_action_in_connection
from .storage_service import StorageSecurityError, read_evidence_bytes, safe_storage_path, verify_checksum


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
    if any(token in text for token in ["ZONE_", "INTRUS", "LOITER", "RUNNING", "PPE_"]):
        return "Security"
    if any(token in text for token in ["ATTENDANCE", "RECOGNITION", "IDENTITY", "SPOOF", "UNKNOWN"]):
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
        "entityId": row.get("entity_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "correlationId": row.get("correlation_id"),
        "eventUid": row.get("event_uid"),
        "sourceFrameId": row.get("source_frame_id"),
        "observationType": row.get("observation_type") or row.get("event_type"),
        "confidence": row.get("confidence") or 0,
        "details": parse_details(row.get("details_json")),
        "message": details_message(row.get("details_json")),
        "snapshot_available": bool(snapshot_path),
        "snapshot_url": f"/api/events/{row['id']}/snapshot" if snapshot_path else None,
        # Never expose absolute local storage paths to browser clients.
        "evidence_available": bool(row.get("evidence_path") or snapshot_path),
        "evidence_path": None,
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
    where = ["e.organization_id=?", "e.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
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
        where e.id = ? and e.organization_id=? and e.site_id=?
        """,
        [event_id, ORGANIZATION_ID, SITE_ID],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    return normalize_event(row)


def event_summary() -> dict[str, Any]:
    rows = fetch_all("select event_type, severity, count(*) as count from events where organization_id=? and site_id=? group by event_type, severity", [ORGANIZATION_ID, SITE_ID])
    security_rows = [
        row for row in rows
        if event_category(str(row.get("event_type") or "")) in {"Security", "Safety", "Object"}
        or "SPOOF" in str(row.get("event_type") or "").upper()
    ]
    return {
        "total": sum(r["count"] for r in rows),
        "by_type": rows,
        "critical": sum(r["count"] for r in rows if int(r["severity"] or 0) >= 3),
        "warning": sum(r["count"] for r in rows if 1 <= int(r["severity"] or 0) < 3),
        # Raw observation volume stays separate from operational incidents.
        "security_total": sum(r["count"] for r in security_rows),
        "security_by_type": security_rows,
    }


def review_event(event_id: int, action: str, note: str | None = None, actor_id: str | None = None) -> dict[str, Any]:
    actions = {"confirm": "confirmed", "dismiss": "dismissed", "escalate": "escalated", "resolve": "resolved"}
    status = actions.get(action)
    if not status:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REVIEW_ACTION", "message": "Use confirm, dismiss, escalate, or resolve."})
    if not fetch_one("select id from events where id=? and organization_id=? and site_id=?", [event_id, ORGANIZATION_ID, SITE_ID]):
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    with transaction(immediate=True) as con:
        con.execute(
            "update events set review_status=?, review_note=?, reviewed_at=datetime('now'), reviewed_by=? where id=? and organization_id=? and site_id=?",
            [status, (note or "").strip()[:500] or None, actor_id or "operator", event_id, ORGANIZATION_ID, SITE_ID],
        )
        record_action_in_connection(
            con,
            "event.review",
            "event",
            event_id,
            {"status": status, "note": note or ""},
            actor_id=actor_id,
        )
    return get_event(event_id)


def snapshot_response(event_id: int) -> Response:
    row = fetch_one("select snapshot_path, evidence_checksum from events where id=? and organization_id=? and site_id=?", [event_id, ORGANIZATION_ID, SITE_ID])
    if not row:
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND", "message": "Event was not found."})
    raw = row.get("snapshot_path")
    if not raw:
        raise HTTPException(status_code=404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "This event has no snapshot."})
    try:
        path = safe_storage_path(raw, base=SNAPSHOTS_DIR)
    except StorageSecurityError:
        raise HTTPException(status_code=400, detail={"code": "INVALID_SNAPSHOT_PATH", "message": "Snapshot path is outside the allowed directory."})
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "Snapshot file is missing."})
    expected = row.get("evidence_checksum")
    if not expected:
        linked = fetch_one(
            "select checksum from incident_evidence where event_id=? and evidence_type='snapshot' order by id desc limit 1",
            [event_id],
        )
        expected = linked.get("checksum") if linked else None
    record_action("evidence.access", "event", event_id, {"result": "requested"}, actor_type="system")
    if expected and not verify_checksum(path, expected):
        with transaction(immediate=True) as con:
            con.execute(
                "update incident_evidence set status='tampered' where event_id=? and evidence_type='snapshot'",
                [event_id],
            )
            record_action_in_connection(
                con,
                "evidence.integrity_failure",
                "event",
                event_id,
                {"result": "checksum_mismatch"},
                actor_type="system",
            )
        raise HTTPException(status_code=409, detail={"code": "EVIDENCE_CHECKSUM_MISMATCH", "message": "Evidence integrity validation failed."})
    try:
        content = read_evidence_bytes(path, expected_checksum=expected)
    except StorageSecurityError as exc:
        with transaction(immediate=True) as con:
            con.execute(
                "update incident_evidence set status='tampered' where event_id=? and evidence_type='snapshot'",
                [event_id],
            )
            record_action_in_connection(
                con,
                "evidence.integrity_failure",
                "event",
                event_id,
                {"result": "authentication_failure", "reason": str(exc)[:160]},
                actor_type="system",
            )
        raise HTTPException(status_code=409, detail={"code": "EVIDENCE_AUTHENTICATION_FAILED", "message": "Evidence authentication failed."}) from exc
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
