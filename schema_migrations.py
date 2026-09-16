"""Small, shared SQLite migrations used by the edge runtime and API.

The project has two database adapters for compatibility. They now share the
same migration ledger and audit-chain format so either adapter can initialize
an existing local database safely.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


AUDIT_TABLES = {
    "audit_log": ("timestamp",),
    "platform_audit_log": ("created_at",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone())


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})").fetchall()}


def _canonical_audit_payload(row: dict[str, Any], previous_hash: str) -> bytes:
    payload = {
        key: value for key, value in row.items()
        if key not in {"record_hash", "prev_hash"}
    }
    payload["prev_hash"] = previous_hash
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def audit_record_hash(row: dict[str, Any], previous_hash: str) -> str:
    return hashlib.sha256(_canonical_audit_payload(row, previous_hash)).hexdigest()


def _backfill_audit_chain(conn: sqlite3.Connection, table: str) -> None:
    if not _table_exists(conn, table):
        return
    rows = conn.execute(f"select * from {table} order by id asc").fetchall()
    previous_hash = ""
    for raw in rows:
        row = dict(raw) if isinstance(raw, sqlite3.Row) else dict(zip(
            [item[1] for item in conn.execute(f"pragma table_info({table})").fetchall()], raw
        ))
        record_hash = audit_record_hash(row, previous_hash)
        conn.execute(
            f"update {table} set prev_hash=?, record_hash=? where id=?",
            (previous_hash, record_hash, row["id"]),
        )
        previous_hash = record_hash


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent storage migrations without dropping existing data."""
    conn.execute(
        """
        create table if not exists schema_migrations (
            version integer primary key,
            description text not null,
            applied_at text not null
        )
        """
    )

    applied = {
        int(row[0]) for row in conn.execute("select version from schema_migrations")
    }
    changed_audit_tables: list[str] = []
    for table in AUDIT_TABLES:
        if not _table_exists(conn, table):
            continue
        columns = _columns(conn, table)
        changed = False
        if "prev_hash" not in columns:
            conn.execute(f"alter table {table} add column prev_hash text")
            changed = True
        if "record_hash" not in columns:
            conn.execute(f"alter table {table} add column record_hash text")
            changed = True
        if changed:
            changed_audit_tables.append(table)

    # Keep the column contract enforced even if a table was created after the
    # ledger's first migration (for example, backend-only boot before the
    # legacy edge database is initialized).
    for table in changed_audit_tables:
        _backfill_audit_chain(conn, table)

    if 1 not in applied:
        for table in AUDIT_TABLES:
            if not _table_exists(conn, table):
                continue
            # Existing rows become the trusted baseline at migration time. Any
            # subsequent modification is detectable through chain validation.
            if table not in changed_audit_tables:
                _backfill_audit_chain(conn, table)

        conn.execute(
            """
            create table if not exists audit_checkpoints (
                id integer primary key autoincrement,
                table_name text not null,
                last_record_id integer not null,
                last_record_hash text not null,
                created_at text not null
            )
            """
        )
        conn.execute(
            "insert into schema_migrations(version, description, applied_at) values (?, ?, ?)",
            (1, "tamper-evident audit chains and shared migration ledger", _utc_now()),
        )
    else:
        conn.execute(
            """
            create table if not exists audit_checkpoints (
                id integer primary key autoincrement,
                table_name text not null,
                last_record_id integer not null,
                last_record_hash text not null,
                created_at text not null
            )
            """
        )


def append_audit_record(
    conn: sqlite3.Connection,
    table: str,
    values: dict[str, Any],
) -> int:
    """Insert and hash one audit row within the caller's transaction."""
    if table not in AUDIT_TABLES:
        raise ValueError(f"Unsupported audit table: {table}")
    if not _table_exists(conn, table):
        raise RuntimeError(f"Audit table does not exist: {table}")
    columns = _columns(conn, table)
    if "prev_hash" not in columns or "record_hash" not in columns:
        raise RuntimeError(f"Audit table is not migrated: {table}")

    previous = conn.execute(
        f"select record_hash from {table} order by id desc limit 1"
    ).fetchone()
    previous_hash = str(previous[0] or "") if previous else ""
    row_values = dict(values)
    row_values["prev_hash"] = previous_hash
    row_values["record_hash"] = None
    names = list(row_values)
    placeholders = ", ".join("?" for _ in names)
    cur = conn.execute(
        f"insert into {table} ({', '.join(names)}) values ({placeholders})",
        [row_values[name] for name in names],
    )
    row_id = int(cur.lastrowid)
    raw = conn.execute(f"select * from {table} where id=?", (row_id,)).fetchone()
    row = dict(raw) if isinstance(raw, sqlite3.Row) else dict(zip(
        [item[1] for item in conn.execute(f"pragma table_info({table})").fetchall()], raw
    ))
    record_hash = audit_record_hash(row, previous_hash)
    conn.execute(f"update {table} set record_hash=? where id=?", (record_hash, row_id))
    return row_id


def verify_audit_chain(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Return a safe integrity report without exposing audit payload secrets."""
    if table not in AUDIT_TABLES or not _table_exists(conn, table):
        return {"ok": False, "table": table, "reason": "missing_table", "checked": 0}
    rows = conn.execute(f"select * from {table} order by id asc").fetchall()
    previous_hash = ""
    for raw in rows:
        row = dict(raw) if isinstance(raw, sqlite3.Row) else dict(zip(
            [item[1] for item in conn.execute(f"pragma table_info({table})").fetchall()], raw
        ))
        if str(row.get("prev_hash") or "") != previous_hash:
            return {"ok": False, "table": table, "reason": "previous_hash_mismatch", "record_id": row.get("id"), "checked": row.get("id", 0)}
        expected = audit_record_hash(row, previous_hash)
        if not row.get("record_hash") or not hmac_compare(str(row["record_hash"]), expected):
            return {"ok": False, "table": table, "reason": "record_hash_mismatch", "record_id": row.get("id"), "checked": row.get("id", 0)}
        previous_hash = str(row["record_hash"])
    return {"ok": True, "table": table, "checked": len(rows), "last_hash": previous_hash}


def hmac_compare(left: str, right: str) -> bool:
    # Kept local to avoid making callers depend on the authentication module.
    import hmac
    return hmac.compare_digest(left, right)
