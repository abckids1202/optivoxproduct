from __future__ import annotations

import json
from typing import Any

from schema_migrations import append_audit_record, create_audit_checkpoint, verify_audit_chain

from ..database import fetch_all, get_connection


def record_action_in_connection(
    con,
    action: str,
    entity_type: str,
    entity_id: Any = None,
    details: dict[str, Any] | None = None,
    actor_type: str = "operator",
    actor_id: str | None = None,
) -> None:
    """Append an audit row to an existing caller-owned transaction.

    Business services use this when an operation and its audit record must be
    committed or rolled back together. The fallback wrapper below remains for
    simple standalone audit actions.
    """
    values = {
        "action": action,
        "entity_type": entity_type,
        "entity_id": None if entity_id is None else str(entity_id),
        "actor_type": actor_type,
        "actor_id": actor_id,
        "details_json": json.dumps(details or {}, sort_keys=True, default=str),
    }
    columns = {row[1] for row in con.execute("pragma table_info(platform_audit_log)").fetchall()}
    # Scope columns are additive and are absent in a few legacy databases.
    # Only include them when the active schema supports them.
    from ..config import DEVICE_ID, ORGANIZATION_ID, SITE_ID
    for column, value in (
        ("organization_id", ORGANIZATION_ID),
        ("site_id", SITE_ID),
        ("device_id", DEVICE_ID),
    ):
        if column in columns:
            values[column] = value
    append_audit_record(con, "platform_audit_log", values)
    create_audit_checkpoint(con, "platform_audit_log")


def record_action(
    action: str,
    entity_type: str,
    entity_id: Any = None,
    details: dict[str, Any] | None = None,
    actor_type: str = "operator",
    actor_id: str | None = None,
) -> None:
    with get_connection() as con:
        # Serialize the read-previous-hash + insert + checkpoint sequence.
        # Without an immediate write lock, concurrent writers could both use
        # the same predecessor and invalidate the audit chain.
        con.execute("BEGIN IMMEDIATE")
        record_action_in_connection(
            con, action, entity_type, entity_id, details, actor_type, actor_id
        )
        con.commit()


def list_actions(limit: int = 100) -> list[dict[str, Any]]:
    return fetch_all("select * from platform_audit_log order by created_at desc, id desc limit ?", [max(1, min(limit, 500))])


def integrity_report() -> dict[str, Any]:
    with get_connection() as con:
        return verify_audit_chain(con, "platform_audit_log")
