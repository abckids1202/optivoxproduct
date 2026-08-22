from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..database import get_connection, fetch_all, fetch_one, execute
from .audit_service import record_action
from .event_service import event_category, severity_label


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
            select id, event_type, severity, timestamp, details_json
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
            incident = con.execute(
                """
                select id, severity from incidents
                where category=? and status not in ('dismissed', 'resolved')
                  and abs(julianday(last_event_at) - julianday(?)) <= (10.0 / 1440.0)
                order by last_event_at desc limit 1
                """,
                [category, row["timestamp"]],
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
                    insert into incidents (status, category, severity, summary, first_event_at, last_event_at)
                    values ('open', ?, ?, ?, ?, ?)
                    """,
                    [category, _severity(row["severity"]), f"{category} activity requires review", row["timestamp"], row["timestamp"]],
                )
                incident_id = cur.lastrowid
            con.execute("insert or ignore into incident_events (incident_id, event_id) values (?, ?)", [incident_id, event_id])
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
    }


def list_incidents(limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
    sync_incidents()
    params: list[Any] = []
    where = ""
    if status:
        where = "where i.status=?"
        params.append(status)
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
    result["events"] = fetch_all(
        """
        select e.*, p.name as person_name
        from incident_events ie join events e on e.id=ie.event_id
        left join people p on p.id=e.person_id
        where ie.incident_id=? order by e.timestamp desc
        """,
        [incident_id],
    )
    return result


def review_incident(incident_id: int, action: str, note: str | None = None) -> dict[str, Any]:
    allowed = {"confirm": "confirmed", "dismiss": "dismissed", "escalate": "escalated", "resolve": "resolved"}
    status = allowed.get(action)
    if not status:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REVIEW_ACTION", "message": "Use confirm, dismiss, escalate, or resolve."})
    if not fetch_one("select id from incidents where id=?", [incident_id]):
        raise HTTPException(status_code=404, detail={"code": "INCIDENT_NOT_FOUND", "message": "Incident was not found."})
    execute(
        "update incidents set status=?, resolution_note=?, updated_at=datetime('now'), resolved_at=case when ? in ('resolved','dismissed') then datetime('now') else resolved_at end, resolved_by=case when ? in ('resolved','dismissed') then 'operator' else resolved_by end where id=?",
        [status, (note or "").strip()[:500] or None, status, status, incident_id],
    )
    record_action("incident.review", "incident", incident_id, {"status": status, "note": note or ""})
    return get_incident(incident_id)
