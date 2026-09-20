from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..config import DEVICE_ID, ORGANIZATION_ID, PROJECT_ROOT, SITE_ID, SNAPSHOTS_DIR
from ..database import get_connection, fetch_all, fetch_one, execute, transaction
from .audit_service import record_action_in_connection
from .event_service import event_category, normalize_event, severity_label
from .outbox_service import enqueue_in_connection
from .storage_service import StorageSecurityError, safe_storage_path, sha256_file, validate_evidence_file, verify_checksum


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
                   snapshot_path, evidence_checksum, source_frame_id,
                   correlation_id
            from events
            where (coalesce(severity, 0) >= 1
               or upper(event_type) like '%SPOOF%'
               or upper(event_type) like '%UNKNOWN%'
               or upper(event_type) like '%DANGER%')
              and organization_id=?
              and site_id=?
            order by timestamp desc limit 500
            """
        , [ORGANIZATION_ID, SITE_ID]).fetchall()
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
            # A correlation ID is useful evidence, but it is not a license to
            # merge unrelated streams. Keep every operational context key in
            # the grouping predicate so a reused/colliding ID cannot join
            # incidents from another entity, camera, session, or zone.
            incident = con.execute(
                """
                select id, severity from incidents
                where category=?
                  and organization_id=?
                  and site_id=?
                  and coalesce(correlation_id, '')=coalesce(?, '')
                  and coalesce(entity_id, '')=coalesce(?, '')
                  and coalesce(camera_id, '')=coalesce(?, '')
                  and coalesce(presence_session_id, 0)=coalesce(?, 0)
                  and coalesce(zone_id, '')=coalesce(?, '')
                  and coalesce(location, '')=coalesce(?, '')
                  and status not in ('dismissed', 'resolved')
                  and abs(julianday(last_event_at) - julianday(?)) <= (10.0 / 1440.0)
                order by last_event_at desc limit 1
                """,
                [category, ORGANIZATION_ID, SITE_ID, row["correlation_id"], row["entity_id"], row["camera_id"],
                 row["presence_session_id"], zone_id, row["location"], row["timestamp"]],
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
                    insert into incidents (status, category, severity, summary, first_event_at, last_event_at, entity_id, camera_id, location, zone_id, presence_session_id, correlation_id, organization_id, site_id, device_id)
                    values ('open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [category, _severity(row["severity"]), f"{category} activity requires review", row["timestamp"], row["timestamp"], row["entity_id"], row["camera_id"], row["location"], zone_id, row["presence_session_id"], row["correlation_id"], ORGANIZATION_ID, SITE_ID, DEVICE_ID],
                )
                incident_id = cur.lastrowid
            con.execute("insert or ignore into incident_events (incident_id, event_id) values (?, ?)", [incident_id, event_id])
            if row["snapshot_path"]:
                raw_path = str(row["snapshot_path"])
                checksum = None
                status = "missing"
                try:
                    # Runtime snapshots normally live under SNAPSHOTS_DIR. A
                    # project-local absolute path remains ingestible for old
                    # deployments and tests, but serving is still restricted
                    # to SNAPSHOTS_DIR by event_service.snapshot_response.
                    evidence_path = safe_storage_path(
                        raw_path,
                        roots=(SNAPSHOTS_DIR, PROJECT_ROOT),
                        base=SNAPSHOTS_DIR,
                    )
                    if evidence_path.is_file():
                        checksum = row["evidence_checksum"] or sha256_file(evidence_path)
                        status = "available" if not row["evidence_checksum"] or verify_checksum(evidence_path, row["evidence_checksum"]) else "tampered"
                        if status == "available":
                            try:
                                validate_evidence_file(
                                    evidence_path,
                                    row["evidence_checksum"],
                                    roots=(SNAPSHOTS_DIR, PROJECT_ROOT),
                                )
                            except StorageSecurityError:
                                status = "tampered"
                    else:
                        status = "missing"
                except StorageSecurityError:
                    evidence_path = None
                    status = "invalid_path"
                con.execute(
                    """insert or ignore into incident_evidence
                       (incident_id, event_id, path, evidence_type, source_frame_id, captured_at, checksum, status)
                       values (?, ?, ?, 'snapshot', ?, ?, ?, ?)""",
                    [incident_id, event_id, raw_path if evidence_path is None else str(evidence_path), row["source_frame_id"],
                     row["timestamp"], checksum, status],
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
        "evidence_count": int(row.get("evidence_count") or 0),
        "alert_count": int(row.get("alert_count") or 0),
        "review_action_count": int(row.get("review_action_count") or 0),
        "resolution_note": row.get("resolution_note"),
        "updated_at": row.get("updated_at"),
        "entityId": row.get("entity_id"),
        "cameraId": row.get("camera_id"),
        "location": row.get("location"),
        "zoneId": row.get("zone_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "correlationId": row.get("correlation_id"),
        "assignedTo": row.get("assigned_to"),
        "falsePositive": bool(row.get("false_positive")),
    }


def _reconcile_incident_evidence(incident_id: int) -> None:
    """Re-check linked evidence before an incident detail is shown.

    Evidence is immutable from the application's point of view. A checksum is
    captured when a runtime event is first materialized; later reads must
    detect replacement, deletion, symlink/path escape, or authenticated
    envelope failure rather than trusting the original status forever.
    """
    with transaction(immediate=True) as con:
        rows = con.execute(
            """
            select id, event_id, path, checksum, status
            from incident_evidence
            where incident_id=?
            order by id
            """,
            [incident_id],
        ).fetchall()
        for row in rows:
            previous_status = str(row["status"] or "available")
            status = "missing"
            failure_reason: str | None = None
            try:
                raw_path = Path(str(row["path"] or ""))
                candidate = raw_path if raw_path.is_absolute() else SNAPSHOTS_DIR / raw_path
                evidence_path = safe_storage_path(
                    row["path"],
                    roots=(SNAPSHOTS_DIR, PROJECT_ROOT),
                    base=SNAPSHOTS_DIR,
                )
                if candidate.is_symlink() or evidence_path.is_symlink():
                    status = "invalid_path"
                    failure_reason = "symlink_not_allowed"
                elif not evidence_path.is_file():
                    status = "missing"
                    failure_reason = "file_missing"
                elif row["checksum"] and not verify_checksum(evidence_path, row["checksum"]):
                    status = "tampered"
                    failure_reason = "checksum_mismatch"
                else:
                    try:
                        validate_evidence_file(
                            evidence_path,
                            row["checksum"],
                            roots=(SNAPSHOTS_DIR, PROJECT_ROOT),
                        )
                    except StorageSecurityError as exc:
                        status = "tampered"
                        failure_reason = str(exc)[:160]
                    else:
                        # Legacy rows without a captured checksum can still be
                        # inspected, but they cannot honestly be called
                        # verified until a trusted baseline exists.
                        status = "available" if row["checksum"] else "unverified"
            except StorageSecurityError as exc:
                status = "invalid_path"
                failure_reason = str(exc)[:160]
            except (OSError, ValueError, TypeError) as exc:
                status = "invalid_path"
                failure_reason = str(exc)[:160]

            if status == previous_status:
                continue
            con.execute(
                "update incident_evidence set status=? where id=?",
                [status, row["id"]],
            )
            if status in {"tampered", "invalid_path"}:
                record_action_in_connection(
                    con,
                    "evidence.integrity_failure",
                    "incident_evidence",
                    row["id"],
                    {
                        "incident_id": incident_id,
                        "event_id": row["event_id"],
                        "status": status,
                        "reason": failure_reason or "integrity_check_failed",
                    },
                    actor_type="system",
                )
            elif status == "missing":
                record_action_in_connection(
                    con,
                    "evidence.unavailable",
                    "incident_evidence",
                    row["id"],
                    {
                        "incident_id": incident_id,
                        "event_id": row["event_id"],
                        "reason": failure_reason or "file_missing",
                    },
                    actor_type="system",
                )


def list_incidents(limit: int = 100, status: str | None = None,
                   category: str | None = None, zone_id: str | None = None,
                   camera_id: str | None = None, severity: int | None = None,
                   assignee: str | None = None) -> list[dict[str, Any]]:
    sync_incidents()
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    filters = ["i.organization_id=?", "i.site_id=?"]
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
            select i.*, count(ie.event_id) as event_count,
                   (select count(*) from incident_evidence x where x.incident_id=i.id) as evidence_count,
                   (select count(*) from incident_alerts x where x.incident_id=i.id) as alert_count,
                   (select count(*) from incident_review_actions x where x.incident_id=i.id) as review_action_count
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
        select i.*, count(ie.event_id) as event_count,
               (select count(*) from incident_evidence x where x.incident_id=i.id) as evidence_count,
               (select count(*) from incident_alerts x where x.incident_id=i.id) as alert_count,
               (select count(*) from incident_review_actions x where x.incident_id=i.id) as review_action_count
        from incidents i left join incident_events ie on ie.incident_id=i.id
            where i.id=? and i.organization_id=? and i.site_id=? group by i.id
        """,
        [incident_id, ORGANIZATION_ID, SITE_ID],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    _reconcile_incident_evidence(incident_id)
    result = _normalize(row)
    result["events"] = [normalize_event(row) for row in fetch_all(
        """
        select e.*, p.name as person_name
        from incident_events ie join events e on e.id=ie.event_id
        left join people p on p.id=e.person_id
        where ie.incident_id=? and e.organization_id=? and e.site_id=? order by e.timestamp desc
        """,
        [incident_id, ORGANIZATION_ID, SITE_ID],
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
    if not fetch_one("select id from incidents where id=? and organization_id=? and site_id=?", [incident_id, ORGANIZATION_ID, SITE_ID]):
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    clean_note = (note or "").strip()[:500] or None
    with transaction(immediate=True) as con:
        con.execute(
            "update incidents set status=?, resolution_note=?, false_positive=case when ?='false_positive' then 1 else false_positive end, updated_at=datetime('now'), resolved_at=case when ? in ('resolved','dismissed') then datetime('now') else resolved_at end, resolved_by=case when ? in ('resolved','dismissed') then ? else resolved_by end where id=? and organization_id=? and site_id=?",
            [status, clean_note, action, status, status, actor_id or "operator", incident_id, ORGANIZATION_ID, SITE_ID],
        )
        con.execute(
            "insert into incident_review_actions (incident_id, action, note, actor_id, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?, ?)",
            [incident_id, action, clean_note, actor_id or "operator", ORGANIZATION_ID, SITE_ID, DEVICE_ID],
        )
        record_action_in_connection(
            con, "incident.review", "incident", incident_id,
            {"status": status, "note": clean_note or "", "action": action},
            actor_id=actor_id,
        )
        enqueue_in_connection(
            con,
            "incident.reviewed",
            {"incident_id": incident_id, "status": status, "action": action},
            aggregate_type="incident",
            aggregate_id=incident_id,
        )
    return get_incident(incident_id)


def assign_incident(incident_id: int, assignee: str | None, actor_id: str | None = None) -> dict[str, Any]:
    if not fetch_one("select id from incidents where id=? and organization_id=? and site_id=?", [incident_id, ORGANIZATION_ID, SITE_ID]):
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    assignee = (assignee or "").strip()[:120] or None
    with transaction(immediate=True) as con:
        con.execute(
            "update incidents set assigned_to=?, updated_at=datetime('now') where id=? and organization_id=? and site_id=?",
            [assignee, incident_id, ORGANIZATION_ID, SITE_ID],
        )
        con.execute(
            "insert into incident_review_actions (incident_id, action, note, actor_id, organization_id, site_id, device_id) values (?, 'assign', ?, ?, ?, ?, ?)",
            [incident_id, assignee, actor_id or "operator", ORGANIZATION_ID, SITE_ID, DEVICE_ID],
        )
        record_action_in_connection(
            con, "incident.assign", "incident", incident_id,
            {"assignee": assignee}, actor_id=actor_id,
        )
        enqueue_in_connection(
            con,
            "incident.assigned",
            {"incident_id": incident_id, "assignee": assignee},
            aggregate_type="incident",
            aggregate_id=incident_id,
        )
    return get_incident(incident_id)
