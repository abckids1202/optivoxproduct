"""Application-security operations for the OptiVox edge and API.

Cybersecurity telemetry is deliberately stored separately from physical
vision events.  This module provides the small operational chain needed by
the dashboard: event -> incident -> alert attempt -> review -> resolution.
It is local-first and never returns session tokens, passwords, or embeddings.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from ..config import DEVICE_ID, HEALTH_EVENTS_PATH
from ..database import fetch_all, fetch_one, get_connection
from .audit_service import list_actions, record_action


CYBER_EVENT_TYPES = frozenset({
    "FAILED_LOGIN_BURST",
    "EXPIRED_TOKEN",
    "WRONG_ROLE_ACCESS",
    "UNAUTHORIZED_COMMAND",
    "BULK_EXPORT",
    "BIOMETRIC_CHANGE",
    "CONFIGURATION_CHANGE",
    "MODEL_INTEGRITY_FAILURE",
    "DATABASE_INTEGRITY_FAILURE",
    "CAMERA_DISCONNECT",
    "RUNTIME_CRASH",
    "ALERT_DELIVERY_FAILURE",
    "UNEXPECTED_OUTBOUND_DESTINATION",
})

SEVERITY_BY_TYPE = {
    "FAILED_LOGIN_BURST": 2,
    "EXPIRED_TOKEN": 1,
    "WRONG_ROLE_ACCESS": 2,
    "UNAUTHORIZED_COMMAND": 3,
    "BULK_EXPORT": 2,
    "BIOMETRIC_CHANGE": 2,
    "CONFIGURATION_CHANGE": 2,
    "MODEL_INTEGRITY_FAILURE": 3,
    "DATABASE_INTEGRITY_FAILURE": 3,
    "CAMERA_DISCONNECT": 2,
    "RUNTIME_CRASH": 3,
    "ALERT_DELIVERY_FAILURE": 3,
    "UNEXPECTED_OUTBOUND_DESTINATION": 3,
}

REVIEW_STATUSES = {
    "acknowledge": "acknowledged",
    "assign": "assigned",
    "confirm": "confirmed",
    "dismiss": "dismissed",
    "false_positive": "dismissed",
    "escalate": "escalated",
    "resolve": "resolved",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any = None) -> str:
    if value is None:
        return _now().isoformat(timespec="seconds")
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return _now().isoformat(timespec="seconds")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {"message": str(value)}


def _text(value: Any, limit: int) -> str | None:
    text = str(value or "").strip()
    return text[:limit] or None


def event_category(event_type: str) -> str:
    if event_type in {"FAILED_LOGIN_BURST", "EXPIRED_TOKEN", "WRONG_ROLE_ACCESS"}:
        return "authentication"
    if event_type == "UNAUTHORIZED_COMMAND":
        return "authorization"
    if event_type in {"BIOMETRIC_CHANGE", "CONFIGURATION_CHANGE", "BULK_EXPORT"}:
        return "administration"
    if event_type in {"MODEL_INTEGRITY_FAILURE", "DATABASE_INTEGRITY_FAILURE"}:
        return "integrity"
    if event_type in {"CAMERA_DISCONNECT", "RUNTIME_CRASH"}:
        return "device_health"
    if event_type in {"ALERT_DELIVERY_FAILURE", "UNEXPECTED_OUTBOUND_DESTINATION"}:
        return "communications"
    return "application_security"


def _summary(event_type: str, details: dict[str, Any]) -> str:
    labels = event_type.replace("_", " ").title()
    subject = details.get("username") or details.get("command") or details.get("camera_id")
    return f"{labels}{f' · {subject}' if subject else ''}"


def _base_group_key(event_type: str, category: str, actor_id: Any, device_id: Any, ip_address: Any, source: str) -> str:
    parts = [category, event_type, _text(actor_id, 120) or "-", _text(device_id, 120) or "-", _text(ip_address, 128) or "-", _text(source, 80) or "-" ]
    return "|".join(parts)


def _existing_open_incident(con, base_key: str, occurred_at: str, window_seconds: int) -> dict[str, Any] | None:
    rows = con.execute(
        "select * from cybersecurity_incidents where correlation_key=? or correlation_key like ? order by id desc",
        (base_key, base_key + "|%"),
    ).fetchall()
    current = datetime.fromisoformat(occurred_at)
    for raw in rows:
        row = dict(raw)
        if row.get("status") in {"resolved", "dismissed"}:
            continue
        try:
            last = datetime.fromisoformat(str(row["last_event_at"]).replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if abs((current - last).total_seconds()) <= max(1, window_seconds):
                return row
        except (TypeError, ValueError):
            continue
    return None


def _unique_incident_key(con, base_key: str, occurred_at: str) -> str:
    if not con.execute("select 1 from cybersecurity_incidents where correlation_key=?", (base_key,)).fetchone():
        return base_key
    return f"{base_key}|{occurred_at}|{uuid.uuid4().hex[:8]}"


def record_cyber_event(
    event_type: str,
    *,
    source: str,
    details: dict[str, Any] | None = None,
    actor_id: Any = None,
    actor_type: str | None = None,
    device_id: Any = None,
    ip_address: Any = None,
    user_agent: Any = None,
    occurred_at: Any = None,
    evidence_ref: Any = None,
    dedupe_key: str | None = None,
    severity: int | None = None,
    group_window_seconds: int = 300,
) -> dict[str, Any]:
    """Record one cyber event and correlate it into a separate incident."""
    event_type = str(event_type or "").strip().upper()
    if event_type not in CYBER_EVENT_TYPES:
        raise ValueError(f"Unsupported cybersecurity event type: {event_type}")
    details = dict(details or {})
    occurred = _iso(occurred_at)
    actor = _text(actor_id, 120)
    device = _text(device_id, 120) or DEVICE_ID
    ip = _text(ip_address, 128)
    ua = _text(user_agent, 300)
    source = _text(source, 80) or "application"
    level = max(0, min(5, int(severity if severity is not None else SEVERITY_BY_TYPE[event_type])))
    category = event_category(event_type)
    dedupe = _text(dedupe_key, 240)
    correlation_id = f"cyber-{uuid.uuid4().hex}"
    log_reference = _text(evidence_ref, 500) or f"audit:cyber.event.recorded:{correlation_id}"
    with get_connection() as con:
        if dedupe:
            existing = con.execute("select * from cybersecurity_events where dedupe_key=?", (dedupe,)).fetchone()
            if existing:
                result = normalize_event(dict(existing))
                result["deduplicated"] = True
                return result
        event_cur = con.execute(
            """insert into cybersecurity_events
               (correlation_id, dedupe_key, event_type, category, severity, source,
                actor_id, actor_type, device_id, ip_address, user_agent, occurred_at,
                details_json, evidence_ref)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (correlation_id, dedupe, event_type, category, level, source, actor,
             _text(actor_type, 40), device, ip, ua, occurred, json.dumps(details, sort_keys=True, default=str), log_reference),
        )
        event_id = int(event_cur.lastrowid)
        base_key = _base_group_key(event_type, category, actor, device, ip, source)
        incident = _existing_open_incident(con, base_key, occurred, group_window_seconds)
        if incident:
            incident_id = int(incident["id"])
            con.execute(
                "update cybersecurity_incidents set severity=max(severity, ?), last_event_at=?, updated_at=datetime('now') where id=?",
                (level, occurred, incident_id),
            )
        else:
            incident_key = _unique_incident_key(con, base_key, occurred)
            incident_cur = con.execute(
                """insert into cybersecurity_incidents
                   (correlation_key, category, severity, summary, source, actor_id,
                    device_id, ip_address, user_agent, first_event_at, last_event_at)
                   values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (incident_key, category, level, _summary(event_type, details), source, actor,
                 device, ip, ua, occurred, occurred),
            )
            incident_id = int(incident_cur.lastrowid)
            con.execute(
                "insert into cybersecurity_incident_alerts (incident_id, channel, status, correlation_id) values (?, 'local_review_queue', 'recorded', ?)",
                (incident_id, correlation_id),
            )
        con.execute("update cybersecurity_events set incident_id=? where id=?", (incident_id, event_id))
        con.execute("insert into cybersecurity_incident_events (incident_id, cyber_event_id) values (?, ?)", (incident_id, event_id))
        con.commit()
    record_action(
        "cyber.event.recorded", "cybersecurity_event", event_id,
        {"event_type": event_type, "incident_id": incident_id, "correlation_id": correlation_id, "source": source},
        actor_type=actor_type or "system", actor_id=actor,
    )
    result = normalize_event(fetch_one("select * from cybersecurity_events where id=?", [event_id]) or {})
    result["deduplicated"] = False
    return result


def normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "correlationId": row.get("correlation_id"),
        "eventType": row.get("event_type"),
        "category": row.get("category"),
        "severity": row.get("severity"),
        "source": row.get("source"),
        "actorId": row.get("actor_id"),
        "actorType": row.get("actor_type"),
        "deviceId": row.get("device_id"),
        "ipAddress": row.get("ip_address"),
        "userAgent": row.get("user_agent"),
        "occurredAt": row.get("occurred_at"),
        "details": _json(row.get("details_json")),
        "evidenceRef": row.get("evidence_ref"),
        "status": row.get("status"),
        "incidentId": row.get("incident_id"),
    }


def normalize_incident(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "correlationKey": row.get("correlation_key"),
        "category": row.get("category"),
        "severity": row.get("severity"),
        "summary": row.get("summary"),
        "source": row.get("source"),
        "actorId": row.get("actor_id"),
        "deviceId": row.get("device_id"),
        "ipAddress": row.get("ip_address"),
        "userAgent": row.get("user_agent"),
        "firstEventAt": row.get("first_event_at"),
        "lastEventAt": row.get("last_event_at"),
        "status": row.get("status"),
        "assignedTo": row.get("assigned_to"),
        "resolutionNote": row.get("resolution_note"),
        "resolvedAt": row.get("resolved_at"),
        "resolvedBy": row.get("resolved_by"),
        "eventCount": int(row.get("event_count") or 0),
        "alertCount": int(row.get("alert_count") or 0),
        "reviewCount": int(row.get("review_count") or 0),
        "updatedAt": row.get("updated_at"),
    }


def _sync_runtime_events() -> int:
    if not HEALTH_EVENTS_PATH.exists():
        return 0
    created = 0
    try:
        lines = HEALTH_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line_number, line in enumerate(lines[-500:], start=max(1, len(lines) - 499)):
        try:
            raw = json.loads(line)
        except (TypeError, ValueError):
            continue
        raw_type = str(raw.get("type") or "").upper()
        state = str(raw.get("state") or "").upper()
        if raw_type == "CAMERA_HEALTH" and state in {"DISCONNECTED", "NO_FRAME", "FROZEN"}:
            event_type = "CAMERA_DISCONNECT"
        elif raw_type == "CONFIGURATION_REJECTED":
            issues = raw.get("issues") or []
            event_type = "MODEL_INTEGRITY_FAILURE" if any("model" in str(item).lower() for item in issues) else "CONFIGURATION_CHANGE"
        elif raw_type == "RUNTIME_CRASH":
            event_type = "RUNTIME_CRASH"
        else:
            continue
        timestamp = raw.get("timestamp") or raw.get("recorded_at")
        dedupe = f"runtime-health:{raw.get('recorded_at')}:{line_number}:{raw_type}:{state}"
        result = record_cyber_event(
            event_type,
            source="runtime",
            device_id=(raw.get("camera_id") or DEVICE_ID),
            occurred_at=timestamp,
            evidence_ref=str(HEALTH_EVENTS_PATH),
            details={"health_type": raw_type, "state": state, "issues": raw.get("issues", []), "process": raw.get("process", {})},
            dedupe_key=dedupe,
        )
        if result.get("id") and not result.get("deduplicated"):
            created += 1
    return created


def _sync_alert_failures() -> int:
    rows = fetch_all("select * from alert_log where lower(coalesce(status, '')) in ('failed', 'error') order by id desc limit 500")
    created = 0
    for row in rows:
        error = str(row.get("error") or "")
        event_type = "UNEXPECTED_OUTBOUND_DESTINATION" if any(term in error.lower() for term in ("webhook", "outbound", "destination", "redirect")) else "ALERT_DELIVERY_FAILURE"
        result = record_cyber_event(
            event_type,
            source="alert_delivery",
            device_id=DEVICE_ID,
            occurred_at=row.get("timestamp"),
            evidence_ref=f"alert_log:{row.get('id')}",
            details={"alert_log_id": row.get("id"), "channel": row.get("channel"), "event_type": row.get("event_type"), "error": error[:500]},
            dedupe_key=f"alert-failure:{row.get('id')}",
        )
        if result.get("id") and not result.get("deduplicated"):
            created += 1
    return created


def sync_operational_security_sources() -> dict[str, int]:
    return {"runtime": _sync_runtime_events(), "alerts": _sync_alert_failures()}


def list_cyber_incidents(limit: int = 100, status: str | None = None, category: str | None = None, severity: int | None = None) -> list[dict[str, Any]]:
    sync_operational_security_sources()
    filters, params = [], []
    if status and status != "all":
        filters.append("i.status=?"); params.append(status)
    if category and category != "all":
        filters.append("i.category=?"); params.append(category)
    if severity is not None:
        filters.append("i.severity=?"); params.append(int(severity))
    where = "where " + " and ".join(filters) if filters else ""
    params.append(max(1, min(int(limit), 500)))
    rows = fetch_all(
        f"""select i.*,
             (select count(*) from cybersecurity_incident_events x where x.incident_id=i.id) as event_count,
             (select count(*) from cybersecurity_incident_alerts x where x.incident_id=i.id) as alert_count,
             (select count(*) from cybersecurity_reviews x where x.incident_id=i.id) as review_count
           from cybersecurity_incidents i {where}
           order by i.updated_at desc, i.id desc limit ?""", params,
    )
    return [normalize_incident(row) for row in rows]


def get_cyber_incident(incident_id: int) -> dict[str, Any]:
    sync_operational_security_sources()
    row = fetch_one(
        """select i.*,
          (select count(*) from cybersecurity_incident_events x where x.incident_id=i.id) as event_count,
          (select count(*) from cybersecurity_incident_alerts x where x.incident_id=i.id) as alert_count,
          (select count(*) from cybersecurity_reviews x where x.incident_id=i.id) as review_count
          from cybersecurity_incidents i where i.id=?""", [incident_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "CYBER_INCIDENT_NOT_FOUND", "message": "Cybersecurity incident was not found."})
    result = normalize_incident(row)
    result["events"] = [normalize_event(item) for item in fetch_all("select e.* from cybersecurity_incident_events ie join cybersecurity_events e on e.id=ie.cyber_event_id where ie.incident_id=? order by e.occurred_at desc, e.id desc", [incident_id])]
    result["alerts"] = fetch_all("select id, incident_id, channel, status, correlation_id, attempted_at, delivered_at, error, attempt_count from cybersecurity_incident_alerts where incident_id=? order by attempted_at desc, id desc", [incident_id])
    result["reviews"] = fetch_all("select id, incident_id, action, note, actor_id, created_at from cybersecurity_reviews where incident_id=? order by created_at desc, id desc", [incident_id])
    return result


def review_cyber_incident(incident_id: int, action: str, note: str | None = None, actor_id: str | None = None) -> dict[str, Any]:
    status = REVIEW_STATUSES.get(str(action or "").strip().lower())
    if not status:
        raise HTTPException(status_code=400, detail={"code": "INVALID_CYBER_REVIEW_ACTION", "message": "Use acknowledge, confirm, dismiss, escalate, or resolve."})
    if not fetch_one("select id from cybersecurity_incidents where id=?", [incident_id]):
        raise HTTPException(status_code=404, detail={"code": "CYBER_INCIDENT_NOT_FOUND", "message": "Cybersecurity incident was not found."})
    clean_note = _text(note, 500)
    execute_sql = "update cybersecurity_incidents set status=?, resolution_note=?, resolved_at=case when ? in ('resolved','dismissed') then datetime('now') else resolved_at end, resolved_by=case when ? in ('resolved','dismissed') then ? else resolved_by end, updated_at=datetime('now') where id=?"
    with get_connection() as con:
        con.execute(execute_sql, (status, clean_note, status, status, actor_id or "operator", incident_id))
        con.execute("insert into cybersecurity_reviews (incident_id, action, note, actor_id) values (?, ?, ?, ?)", (incident_id, action, clean_note, actor_id or "operator"))
        con.commit()
    record_action("cyber.incident.review", "cybersecurity_incident", incident_id, {"action": action, "status": status, "note": clean_note or ""}, actor_id=actor_id)
    return get_cyber_incident(incident_id)


def assign_cyber_incident(incident_id: int, assignee: str | None, actor_id: str | None = None) -> dict[str, Any]:
    if not fetch_one("select id from cybersecurity_incidents where id=?", [incident_id]):
        raise HTTPException(status_code=404, detail={"code": "CYBER_INCIDENT_NOT_FOUND", "message": "Cybersecurity incident was not found."})
    assignee = _text(assignee, 120)
    with get_connection() as con:
        con.execute("update cybersecurity_incidents set assigned_to=?, status=case when status='open' and ? is not null then 'assigned' else status end, updated_at=datetime('now') where id=?", (assignee, assignee, incident_id))
        con.execute("insert into cybersecurity_reviews (incident_id, action, note, actor_id) values (?, 'assign', ?, ?)", (incident_id, assignee, actor_id or "operator"))
        con.commit()
    record_action("cyber.incident.assign", "cybersecurity_incident", incident_id, {"assignee": assignee}, actor_id=actor_id)
    return get_cyber_incident(incident_id)


def list_cyber_events(limit: int = 200, event_type: str | None = None) -> list[dict[str, Any]]:
    sync_operational_security_sources()
    if event_type:
        rows = fetch_all("select * from cybersecurity_events where event_type=? order by occurred_at desc, id desc limit ?", [event_type, max(1, min(int(limit), 500))])
    else:
        rows = fetch_all("select * from cybersecurity_events order by occurred_at desc, id desc limit ?", [max(1, min(int(limit), 500))])
    return [normalize_event(row) for row in rows]


def cybersecurity_summary() -> dict[str, Any]:
    sync_operational_security_sources()
    statuses = fetch_all("select status, count(*) as count from cybersecurity_incidents group by status")
    categories = fetch_all("select category as name, count(*) as value from cybersecurity_incidents group by category order by value desc")
    types = fetch_all("select event_type as name, count(*) as value from cybersecurity_events group by event_type order by value desc")
    return {
        "incidentTotal": sum(int(row["count"]) for row in statuses),
        "openIncidents": sum(int(row["count"]) for row in statuses if row["status"] not in {"dismissed", "resolved"}),
        "eventTotal": sum(int(row["value"]) for row in types),
        "statusCounts": statuses,
        "categories": categories,
        "eventTypes": types,
        "physicalEventsSeparate": True,
        "lastEvent": (list_cyber_events(1) or [None])[0],
    }


def active_sessions() -> list[dict[str, Any]]:
    return fetch_all("""select s.user_id, u.username, u.role, s.client_ip, s.user_agent,
        s.created_at, s.last_seen_at, s.expires_at
        from platform_sessions s join platform_users u on u.id=s.user_id
        where u.active=1 and s.expires_at>? order by s.last_seen_at desc""", [_now().isoformat()])


def authentication_history(limit: int = 100) -> list[dict[str, Any]]:
    rows = [row for row in list_actions(limit=max(1, min(limit * 2, 500))) if str(row.get("action") or "").startswith("auth.")]
    for row in rows:
        row["details"] = _json(row.get("details_json"))
        row.pop("details_json", None)
    return rows[:max(1, min(limit, 500))]


def administrative_history(limit: int = 100) -> list[dict[str, Any]]:
    rows = [row for row in list_actions(limit=max(1, min(limit * 3, 500))) if not str(row.get("action") or "").startswith(("auth.", "cyber."))]
    for row in rows:
        row["details"] = _json(row.get("details_json"))
        row.pop("details_json", None)
    return rows[:max(1, min(limit, 500))]


def integrity_failures(limit: int = 100) -> list[dict[str, Any]]:
    return [item for item in list_cyber_events(limit=limit) if item["eventType"] in {"MODEL_INTEGRITY_FAILURE", "DATABASE_INTEGRITY_FAILURE"}]


def alert_failures(limit: int = 100) -> list[dict[str, Any]]:
    sync_operational_security_sources()
    return [item for item in list_cyber_events(limit=limit) if item["eventType"] in {"ALERT_DELIVERY_FAILURE", "UNEXPECTED_OUTBOUND_DESTINATION"}]
