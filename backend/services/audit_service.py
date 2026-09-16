from __future__ import annotations

import json
from typing import Any

from schema_migrations import append_audit_record, verify_audit_chain

from ..database import fetch_all, get_connection


def record_action(
    action: str,
    entity_type: str,
    entity_id: Any = None,
    details: dict[str, Any] | None = None,
    actor_type: str = "operator",
    actor_id: str | None = None,
) -> None:
    with get_connection() as con:
        append_audit_record(
            con,
            "platform_audit_log",
            {
                "action": action,
                "entity_type": entity_type,
                "entity_id": None if entity_id is None else str(entity_id),
                "actor_type": actor_type,
                "actor_id": actor_id,
                "details_json": json.dumps(details or {}, sort_keys=True, default=str),
            },
        )
        con.commit()


def list_actions(limit: int = 100) -> list[dict[str, Any]]:
    return fetch_all("select * from platform_audit_log order by created_at desc, id desc limit ?", [max(1, min(limit, 500))])


def integrity_report() -> dict[str, Any]:
    with get_connection() as con:
        return verify_audit_chain(con, "platform_audit_log")
