"""Transactional operational outbox for the local OptiVox database.

The edge runtime remains useful without a network connection. When a control
plane is added, this table provides the durable hand-off between a committed
domain change and an eventual authenticated sync attempt. Payloads are
minimized by rejecting biometric, credential, raw-frame, and local plate data.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import DEVICE_ID, ORGANIZATION_ID, SITE_ID
from ..database import fetch_all, fetch_one, transaction


MAX_PAYLOAD_BYTES = 256 * 1024
MAX_ERROR_LENGTH = 1000
MAX_ATTEMPTS = 8
DEFAULT_LEASE_SECONDS = 300
_FORBIDDEN_KEYS = {
    "embedding", "embeddings", "face_embedding", "password", "password_hash",
    "api_key", "token", "access_token", "refresh_token", "raw_frame",
    "frame_bytes", "image_bytes", "plate_text", "plate_text_local",
}


class OutboxSecurityError(ValueError):
    """Raised when an event would move data that must remain on the edge."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                return str(key)
            found = _forbidden_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _forbidden_key(child)
            if found:
                return found
    return None


def _validate_payload(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise OutboxSecurityError("Outbox payload must be a JSON object.")
    forbidden = _forbidden_key(payload)
    if forbidden:
        raise OutboxSecurityError(f"Sensitive field '{forbidden}' cannot enter the database outbox.")
    encoded = _canonical(payload)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise OutboxSecurityError("Outbox payload exceeds the configured size limit.")
    return encoded


def _scope_where(alias: str = "") -> tuple[str, list[str]]:
    prefix = f"{alias}." if alias else ""
    return (
        f"{prefix}organization_id=? and {prefix}site_id=? and {prefix}device_id=?",
        [ORGANIZATION_ID, SITE_ID, DEVICE_ID],
    )


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    try:
        result["payload"] = json.loads(result.pop("payload_json"))
    except (TypeError, ValueError, json.JSONDecodeError):
        result["payload"] = None
    result["attempts"] = int(result.get("attempts") or 0)
    return result


def enqueue_in_connection(
    con,
    event_type: str,
    payload: dict[str, Any],
    *,
    event_id: str | None = None,
    aggregate_type: str | None = None,
    aggregate_id: str | int | None = None,
    organization_id: str | None = None,
    site_id: str | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    """Insert an idempotent outbox event into an existing transaction."""
    event_type = str(event_type or "").strip()[:160]
    if not event_type:
        raise ValueError("event_type is required")
    payload_json = _validate_payload(payload)
    organization_id = str(organization_id or ORGANIZATION_ID)
    site_id = str(site_id or SITE_ID)
    device_id = str(device_id or DEVICE_ID)
    stable_id = str(event_id or "").strip() or hashlib.sha256(
        f"{event_type}:{payload_json}".encode("utf-8")
    ).hexdigest()
    checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    con.execute(
        """
        insert into platform_outbox
            (event_id, event_type, aggregate_type, aggregate_id, payload_json,
             payload_checksum, organization_id, site_id, device_id)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(event_id) do nothing
        """,
        (
            stable_id, event_type, (str(aggregate_type)[:120] if aggregate_type else None),
            (str(aggregate_id)[:160] if aggregate_id is not None else None),
            payload_json, checksum, organization_id, site_id, device_id,
        ),
    )
    row = con.execute(
        "select * from platform_outbox where event_id=? and organization_id=? and site_id=? and device_id=?",
        (stable_id, organization_id, site_id, device_id),
    ).fetchone()
    if not row:
        raise RuntimeError("Outbox insert could not be read back.")
    return _normalize(dict(row))


def enqueue(
    event_type: str,
    payload: dict[str, Any],
    *,
    event_id: str | None = None,
    aggregate_type: str | None = None,
    aggregate_id: str | int | None = None,
    organization_id: str | None = None,
    site_id: str | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    with transaction(immediate=True) as con:
        return enqueue_in_connection(
            con, event_type, payload, event_id=event_id,
            aggregate_type=aggregate_type, aggregate_id=aggregate_id,
            organization_id=organization_id, site_id=site_id, device_id=device_id,
        )


def _recover_stale_claims_in_connection(con, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> int:
    """Return abandoned processing rows to retryable state in this transaction."""
    lease_seconds = max(30, min(int(lease_seconds), 86400))
    where, params = _scope_where()
    result = con.execute(
        f"""update platform_outbox set status='failed', next_attempt_at=datetime('now'),
                locked_at=null, locked_by=null,
                last_error='worker lease expired', updated_at=datetime('now')
            where {where} and status='processing'
              and locked_at is not null
              and locked_at < datetime('now', ?)""",
        [*params, f"-{lease_seconds} seconds"],
    )
    return int(result.rowcount or 0)


def recover_stale_claims(lease_seconds: int = DEFAULT_LEASE_SECONDS) -> int:
    """Recover claims abandoned by a crashed or disconnected sender."""
    with transaction(immediate=True) as con:
        return _recover_stale_claims_in_connection(con, lease_seconds)


def claim_pending(limit: int = 50, worker_id: str = "outbox-worker") -> list[dict[str, Any]]:
    """Atomically claim due events so two workers cannot send the same item."""
    limit = max(1, min(int(limit), 500))
    worker_id = str(worker_id or "outbox-worker")[:120]
    with transaction(immediate=True) as con:
        _recover_stale_claims_in_connection(con)
        where, params = _scope_where()
        rows = con.execute(
            f"""select id from platform_outbox
                where {where} and status in ('pending', 'failed')
                  and next_attempt_at <= datetime('now')
                order by id limit ?""",
            [*params, limit],
        ).fetchall()
        claimed: list[dict[str, Any]] = []
        for row in rows:
            updated = con.execute(
                f"""update platform_outbox set status='processing', locked_at=datetime('now'),
                    locked_by=?, updated_at=datetime('now')
                    where id=? and {where} and status in ('pending', 'failed')""",
                [worker_id, row["id"], *params],
            )
            if updated.rowcount:
                item = con.execute("select * from platform_outbox where id=?", [row["id"]]).fetchone()
                if item:
                    claimed.append(_normalize(dict(item)))
        return claimed


def acknowledge(event_id: str, worker_id: str | None = None) -> bool:
    clauses = ["event_id=?", "status='processing'", "organization_id=?", "site_id=?", "device_id=?"]
    params: list[Any] = [str(event_id), ORGANIZATION_ID, SITE_ID, DEVICE_ID]
    if worker_id:
        clauses.append("locked_by=?")
        params.append(str(worker_id)[:120])
    with transaction(immediate=True) as con:
        result = con.execute(
            f"update platform_outbox set status='sent', sent_at=datetime('now'), updated_at=datetime('now'), last_error=null where {' and '.join(clauses)}",
            params,
        )
        return bool(result.rowcount)


def fail(event_id: str, error: str, worker_id: str | None = None) -> dict[str, Any] | None:
    clauses = ["event_id=?", "status='processing'", "organization_id=?", "site_id=?", "device_id=?"]
    params: list[Any] = [str(event_id), ORGANIZATION_ID, SITE_ID, DEVICE_ID]
    if worker_id:
        clauses.append("locked_by=?")
        params.append(str(worker_id)[:120])
    with transaction(immediate=True) as con:
        row = con.execute(
            f"select * from platform_outbox where {' and '.join(clauses)}",
            params,
        ).fetchone()
        if not row:
            return None
        attempts = int(row["attempts"] or 0) + 1
        status = "dead_letter" if attempts >= MAX_ATTEMPTS else "failed"
        delay = min(3600, int(math.pow(2, min(attempts, 12))))
        next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        con.execute(
            "update platform_outbox set status=?, attempts=?, next_attempt_at=?, last_error=?, locked_at=null, locked_by=null, updated_at=datetime('now') where id=?",
            (status, attempts, next_attempt, str(error or "unknown failure")[:MAX_ERROR_LENGTH], row["id"]),
        )
        updated = con.execute("select * from platform_outbox where id=?", [row["id"]]).fetchone()
        return _normalize(dict(updated)) if updated else None


def requeue_dead_letter(event_id: str, actor_id: str | None = None) -> bool:
    """Return one dead-letter event to the pending queue with an audit row."""
    from .audit_service import record_action_in_connection

    with transaction(immediate=True) as con:
        where, params = _scope_where()
        result = con.execute(
            f"""update platform_outbox set status='pending', attempts=0,
                    next_attempt_at=datetime('now'), locked_at=null, locked_by=null,
                    last_error=null, updated_at=datetime('now')
                where event_id=? and {where} and status='dead_letter'""",
            [str(event_id), *params],
        )
        if not result.rowcount:
            return False
        record_action_in_connection(
            con,
            "outbox.requeue",
            "platform_outbox",
            event_id,
            {"event_id": str(event_id)},
            actor_id=actor_id,
        )
        return True


def purge_sent(retention_days: int, *, execute_delete: bool = False, actor_id: str | None = None) -> dict[str, Any]:
    """Preview or delete only successfully delivered events past retention."""
    from .audit_service import record_action_in_connection

    days = max(1, min(int(retention_days), 3650))
    where, params = _scope_where()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with transaction(immediate=True) as con:
        count_row = con.execute(
            f"select count(*) as count from platform_outbox where {where} and status='sent' and sent_at is not null and sent_at < ?",
            [*params, cutoff],
        ).fetchone()
        count = int(count_row["count"] or 0)
        deleted = 0
        if execute_delete and count:
            result = con.execute(
                f"delete from platform_outbox where {where} and status='sent' and sent_at is not null and sent_at < ?",
                [*params, cutoff],
            )
            deleted = int(result.rowcount or 0)
            record_action_in_connection(
                con,
                "outbox.purge_sent",
                "platform_outbox",
                None,
                {"retention_days": days, "deleted": deleted},
                actor_id=actor_id,
            )
        return {"eligible": count, "deleted": deleted, "retention_days": days, "dry_run": not execute_delete}


def summary() -> dict[str, Any]:
    where, params = _scope_where()
    rows = fetch_all(
        f"select status, count(*) as count from platform_outbox where {where} group by status",
        params,
    )
    counts = {str(row["status"]): int(row["count"]) for row in rows}
    oldest = fetch_one(
        f"select min(created_at) as oldest_pending from platform_outbox where {where} and status in ('pending','failed')",
        params,
    )
    return {
        "status": "healthy" if counts.get("dead_letter", 0) == 0 else "degraded",
        "counts": counts,
        "pending": counts.get("pending", 0) + counts.get("failed", 0),
        "oldest_pending": oldest.get("oldest_pending") if oldest else None,
        "dead_letter": counts.get("dead_letter", 0),
    }
