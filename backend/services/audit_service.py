from __future__ import annotations

import json
from typing import Any

from ..database import execute, fetch_all


def record_action(
    action: str,
    entity_type: str,
    entity_id: Any = None,
    details: dict[str, Any] | None = None,
    actor_type: str = "operator",
    actor_id: str | None = None,
) -> None:
    execute(
        """
        insert into platform_audit_log
            (action, entity_type, entity_id, actor_type, actor_id, details_json)
        values (?, ?, ?, ?, ?, ?)
        """,
        [action, entity_type, None if entity_id is None else str(entity_id), actor_type, actor_id, json.dumps(details or {}, default=str)],
    )


def list_actions(limit: int = 100) -> list[dict[str, Any]]:
    return fetch_all("select * from platform_audit_log order by created_at desc, id desc limit ?", [max(1, min(limit, 500))])
