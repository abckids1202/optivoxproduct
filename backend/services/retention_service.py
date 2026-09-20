"""Policy-aware retention for low-level operational telemetry.

Attendance, roster, incident, evidence metadata, and audit history are
deliberately protected here. Retention must be explicit and reviewable; this
module never deletes data merely because a scheduled task happened to run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..database import get_connection, transaction


MAX_RETENTION_DAYS = 3650
DEFAULT_RETENTION_DAYS = 365

_POLICIES = {
    "events": {
        "time_column": "timestamp",
        "guard": "not exists (select 1 from incident_events x where x.event_id=events.id) and not exists (select 1 from incident_evidence e where e.event_id=events.id)",
    },
    "alert_log": {
        "time_column": "timestamp",
        "guard": "source_event_id is null",
    },
    "cybersecurity_events": {
        "time_column": "occurred_at",
        "guard": "incident_id is null and not exists (select 1 from cybersecurity_incident_events x where x.cyber_event_id=cybersecurity_events.id)",
    },
    "edge_sync_events": {
        "time_column": "received_at",
        "guard": "1=1",
    },
    "edge_sync_batches": {
        "time_column": "received_at",
        "guard": "1=1",
    },
}


def _cutoff(days: int) -> str:
    bounded = max(1, min(int(days), MAX_RETENTION_DAYS))
    return (datetime.now(timezone.utc) - timedelta(days=bounded)).isoformat()


def _table_exists(con, table: str) -> bool:
    return bool(con.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone())


def _row_summary(con, table: str, policy: dict[str, str], cutoff: str) -> dict[str, Any]:
    column = policy["time_column"]
    guard = policy["guard"]
    count = con.execute(
        f"select count(*) from {table} where julianday({column}) < julianday(?) and {guard}",
        (cutoff,),
    ).fetchone()[0]
    oldest = con.execute(
        f"select min({column}) from {table} where julianday({column}) < julianday(?) and {guard}",
        (cutoff,),
    ).fetchone()[0]
    return {"table": table, "eligible": int(count or 0), "oldest_eligible": oldest, "cutoff": cutoff}


def preview(days: int = DEFAULT_RETENTION_DAYS) -> dict[str, Any]:
    """Count purge candidates without mutating the database."""
    bounded = max(1, min(int(days), MAX_RETENTION_DAYS))
    cutoff = _cutoff(bounded)
    with get_connection() as con:
        items = [
            _row_summary(con, table, policy, cutoff)
            for table, policy in _POLICIES.items()
            if _table_exists(con, table)
        ]
    return {
        "dry_run": True,
        "retention_days": bounded,
        "protected": ["people", "attendance", "absence_records", "incidents", "incident_evidence", "platform_audit_log", "audit_log"],
        "items": items,
        "eligible": sum(item["eligible"] for item in items),
    }


def purge(days: int = DEFAULT_RETENTION_DAYS, *, actor_id: str | None = None) -> dict[str, Any]:
    """Delete only unlinked low-level telemetry inside one audited transaction."""
    bounded = max(1, min(int(days), MAX_RETENTION_DAYS))
    cutoff = _cutoff(bounded)
    with transaction(immediate=True) as con:
        items = [
            _row_summary(con, table, policy, cutoff)
            for table, policy in _POLICIES.items()
            if _table_exists(con, table)
        ]
        deleted: dict[str, int] = {}
        # Events are removed before their standalone sync receipts so linked
        # incident guards are evaluated against the intact pre-purge graph.
        for item in items:
            table = item["table"]
            policy = _POLICIES[table]
            result = con.execute(
                f"delete from {table} where julianday({policy['time_column']}) < julianday(?) and {policy['guard']}",
                (cutoff,),
            )
            deleted[table] = int(result.rowcount or 0)
        from .audit_service import record_action_in_connection
        record_action_in_connection(
            con,
            "database.retention_purge",
            "database",
            None,
            {"retention_days": bounded, "cutoff": cutoff, "deleted": deleted},
            actor_type="operator" if actor_id else "system",
            actor_id=actor_id,
        )
        return {
            "dry_run": False,
            "retention_days": bounded,
            "cutoff": cutoff,
            "deleted": deleted,
            "deleted_total": sum(deleted.values()),
            "protected": ["people", "attendance", "absence_records", "incidents", "incident_evidence", "platform_audit_log", "audit_log"],
        }
