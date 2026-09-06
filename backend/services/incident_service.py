from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..database import get_connection, fetch_all, fetch_one, execute
from .audit_service import record_action
from .event_service import event_category, normalize_event, severity_label


def _severity(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def sync_incidents() -> None:
    """Group nearby warning/critical observations into reviewable incidents."""
    with get_connection() as con:
        rows = con.execute(
            """
            select id, event_type, severity, timestamp, details_json,
                   entity_id, presence_session_id, camera_id, location,
                   snapshot_path, source_frame_id
            from events
            where coalesce(severity, 0) >= 1
               or upper(event_type) like '%SPOOF%'
               or upper(event_type) like '%UNKNOWN%'
               or upper(event_type) like '%DANGER%'
            order by timestamp desc limit 500
            """
        ).fetchall()
        for row in rows:
            event_id = row["id"]
            linked = con.execute("select 1 from incident_events where event_id=?", [event_id]).fetchone()
            if linked:
                continue
            category = event_category(row["event_type"] or "")
            try:
                details = json.loads(row["details_json"] or "{}")
            except (TypeError, ValueError):
                details = {}
            security_metadata = details.get("security_metadata") if isinstance(details, dict) else {}
            zone_id = (security_metadata or {}).get("zone_id") if isinstance(security_metadata, dict) else None
            incident = con.execute(
                """
                select id, severity from incidents
                where category=? and coalesce(entity_id, '')=coalesce(?, '')
                  and coalesce(camera_id, '')=coalesce(?, '')
                  and coalesce(presence_session_id, 0)=coalesce(?, 0)
                  and coalesce(zone_id, '')=coalesce(?, '')
                  and coalesce(location, '')=coalesce(?, '')
                  and status not in ('dismissed', 'resolved')
                  and abs(julianday(last_event_at) - julianday(?)) <= (10.0 / 1440.0)
                order by last_event_at desc limit 1
                """,
                [category, row["entity_id"], row["camera_id"], row["presence_session_id"],
                 zone_id, row["location"], row["timestamp"]],
            ).fetchone()
            if incident:
                incident_id = incident["id"]
                con.execute(
                    """
                    update incidents set severity=?, last_event_at=case when julianday(last_event_at) > julianday(?) then last_event_at else ? end, updated_at=datetime('now')
                    where id=?
                    """,
                    [max(_severity(incident["severity"]), _severity(row["severity"])), row["timestamp"], row["timestamp"], incident_id],
                )
            else:
                cur = con.execute(
                    """
                    insert into incidents (status, category, severity, summary, first_event_at, last_event_at, entity_id, camera_id, location, zone_id, presence_session_id)
                    values ('open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [category, _severity(row["severity"]), f"{category} activity requires review", row["timestamp"], row["timestamp"], row["entity_id"], row["camera_id"], row["location"], zone_id, row["presence_session_id"]],
                )
                incident_id = cur.lastrowid
            con.execute("insert or ignore into incident_events (incident_id, event_id) values (?, ?)", [incident_id, event_id])
            if row["snapshot_path"]:
                evidence_path = Path(str(row["snapshot_path"]))
                checksum = None
                try:
                    checksum = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                except OSError:
                    pass
                con.execute(
                    """insert or ignore into incident_evidence
                       (incident_id, event_id, path, evidence_type, source_frame_id, captured_at, checksum, status)
                       values (?, ?, ?, 'snapshot', ?, ?, ?, ?)""",
                    [incident_id, event_id, str(evidence_path), row["source_frame_id"],
                     row["timestamp"], checksum, "available" if evidence_path.exists() else "missing"],
                )
            # Alert delivery is recorded separately from the incident decision.
            # Match the runtime's alert log to the source observation by type and
            # a narrow timestamp window, then make the link idempotent.
            if con.execute("select 1 from sqlite_master where type='table' and name='alert_log'").fetchone():
                alert_rows = con.execute(
                    """
                    select id, channel, status, error, timestamp
                    from alert_log
                    where (source_event_id=? or (source_event_id is null and event_type=?
                           and abs(julianday(timestamp)-julianday(?)) <= (2.0 / 1440.0)))
                    """,
                    [event_id, row["event_type"], row["timestamp"]],
                ).fetchall()
                for alert in alert_rows:
                    con.execute(
                        """
                        insert into incident_alerts
                            (incident_id, channel, status, attempted_at, delivered_at, error, source_alert_id)
                        select ?, ?, ?, ?, case when lower(coalesce(?, '')) in ('sent','delivered','success') then ? else null end, ?, ?
                        where not exists (select 1 from incident_alerts where incident_id=? and source_alert_id=? )
                        """,
                        [incident_id, alert["channel"], alert["status"] or "unknown", alert["timestamp"],
                         alert["status"] or "unknown", alert["timestamp"], alert["error"],
                         alert["id"], incident_id, alert["id"]],
                    )
        con.commit()


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "status": row["status"],
        "category": row["category"],
        "severity": severity_label(row.get("severity")),
        "summary": row["summary"],
        "first_event_at": row["first_event_at"],
        "last_event_at": row["last_event_at"],
        "event_count": int(row.get("event_count") or 0),
        "resolution_note": row.get("resolution_note"),
        "updated_at": row.get("updated_at"),
        "entityId": row.get("entity_id"),
        "cameraId": row.get("camera_id"),
        "location": row.get("location"),
        "zoneId": row.get("zone_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "assignedTo": row.get("assigned_to"),
        "falsePositive": bool(row.get("false_positive")),
    }


def list_incidents(limit: int = 100, status: str | None = None,
                   category: str | None = None, zone_id: str | None = None,
                   camera_id: str | None = None, severity: int | None = None,
                   assignee: str | None = None) -> list[dict[str, Any]]:
    sync_incidents()
    params: list[Any] = []
    filters = []
    for column, value in (
        ("i.status", status), ("i.category", category),
        ("i.zone_id", zone_id), ("i.camera_id", camera_id),
        ("i.assigned_to", assignee),
    ):
        if value:
            filters.append(f"{column}=?")
            params.append(value)
    if severity is not None:
        filters.append("i.severity=?")
        params.append(int(severity))
    where = "where " + " and ".join(filters) if filters else ""
    params.append(max(1, min(limit, 500)))
    return [
        _normalize(row)
        for row in fetch_all(
            f"""
            select i.*, count(ie.event_id) as event_count
            from incidents i left join incident_events ie on ie.incident_id=i.id
            {where}
            group by i.id order by i.updated_at desc, i.id desc limit ?
            """,
            params,
        )
    ]


def get_incident(incident_id: int) -> dict[str, Any]:
    sync_incidents()
    row = fetch_one(
        """
        select i.*, count(ie.event_id) as event_count
        from incidents i left join incident_events ie on ie.incident_id=i.id
        where i.id=? group by i.id
        """,
        [incident_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    result = _normalize(row)
    result["events"] = [normalize_event(row) for row in fetch_all(
        """
        select e.*, p.name as person_name
        from incident_events ie join events e on e.id=ie.event_id
        left join people p on p.id=e.person_id
        where ie.incident_id=? order by e.timestamp desc
        """,
        [incident_id],
    )]
    result["review_actions"] = fetch_all(
        "select * from incident_review_actions where incident_id=? order by created_at desc, id desc",
        [incident_id],
    )
    result["alerts"] = fetch_all(
        "select * from incident_alerts where incident_id=? order by attempted_at desc, id desc",
        [incident_id],
    )
    result["evidence"] = fetch_all(
        "select * from incident_evidence where incident_id=? order by captured_at desc, id desc",
        [incident_id],
    )
    return result


def review_incident(incident_id: int, action: str, note: str | None = None, actor_id: str | None = None) -> dict[str, Any]:
    allowed = {
        "acknowledge": "acknowledged",
        "assign": "assigned",
        "confirm": "confirmed",
        "dismiss": "dismissed",
        "false_positive": "dismissed",
        "escalate": "escalated",
        "resolve": "resolved",
    }
    status = allowed.get(action)
    if not status:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REVIEW_ACTION", "message": "Use acknowledge, assign, confirm, dismiss, false_positive, escalate, or resolve."})
    if not fetch_one("select id from incidents where id=?", [incident_id]):
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    execute(
        "update incidents set status=?, resolution_note=?, false_positive=case when ?='false_positive' then 1 else false_positive end, updated_at=datetime('now'), resolved_at=case when ? in ('resolved','dismissed') then datetime('now') else resolved_at end, resolved_by=case when ? in ('resolved','dismissed') then ? else resolved_by end where id=?",
        [status, (note or "").strip()[:500] or None, action, status, status, actor_id or "operator", incident_id],
    )
    execute(
        "insert into incident_review_actions (incident_id, action, note, actor_id) values (?, ?, ?, ?)",
        [incident_id, action, (note or "").strip()[:500] or None, actor_id or "operator"],
    )
    record_action("incident.review", "incident", incident_id, {"status": status, "note": note or "", "action": action}, actor_id=actor_id)
    return get_incident(incident_id)


def assign_incident(incident_id: int, assignee: str | None, actor_id: str | None = None) -> dict[str, Any]:
    if not fetch_one("select id from incidents where id=?", [incident_id]):
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    assignee = (assignee or "").strip()[:120] or None
    execute("update incidents set assigned_to=?, updated_at=datetime('now') where id=?", [assignee, incident_id])
    execute(
        "insert into incident_review_actions (incident_id, action, note, actor_id) values (?, 'assign', ?, ?)",
        [incident_id, assignee, actor_id or "operator"],
    )
    record_action("incident.assign", "incident", incident_id, {"assignee": assignee}, actor_id=actor_id)
    return get_incident(incident_id)
