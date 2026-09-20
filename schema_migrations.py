"""Small, shared SQLite migrations used by the edge runtime and API.

The project has two database adapters for compatibility. They now share the
same migration ledger and audit-chain format so either adapter can initialize
an existing local database safely.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import socket
import uuid
from datetime import datetime, timezone
from typing import Any


AUDIT_TABLES = {
    "audit_log": ("timestamp",),
    "platform_audit_log": ("created_at",),
}

EXPECTED_SCHEMA_STATES = {
    "edge": (2, hashlib.sha256(b"optivox-edge-schema-v2-scope").hexdigest()),
    "platform": (17, hashlib.sha256(b"optivox-platform-schema-v17-policy-snapshot-registry").hexdigest()),
    "outbox": (3, hashlib.sha256(b"optivox-outbox-schema-v3-immutable-deployment-scope").hexdigest()),
}
EXPECTED_MIGRATION_CHECKSUMS = {
    1: hashlib.sha256(b"optivox-audit-chain-migration-v1").hexdigest(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone())


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})").fetchall()}


def _normalise_sql(sql: str | None) -> str:
    """Normalise SQLite DDL while ignoring deployment-specific defaults.

    The edge and API adapters intentionally use different local device
    defaults. Those values are deployment configuration, not schema shape, so
    they must not make an otherwise identical contract look different.
    """
    value = str(sql or "").lower()
    value = re.sub(r"\bdefault\s+(?:'[^']*'|\"[^\"]*\"|\([^)]*\)|[^,\s)]+)", "default ?", value)
    return re.sub(r"\s+", " ", value).strip()


def schema_fingerprint(conn: sqlite3.Connection, schema_name: str) -> str | None:
    """Return a structural fingerprint for a protected schema contract.

    The edge and platform adapters have separate contracts because they can
    initialize in different orders. Unknown schemas return ``None`` so the
    migration ledger remains extensible without pretending to validate
    contracts that have not been defined.
    """
    if schema_name == "outbox":
        tables_to_fingerprint = ("platform_outbox",)
    elif schema_name == "platform":
        tables_to_fingerprint = (
            "people", "events", "alert_log", "attendance", "policy_snapshots", "absence_records",
            "attendance_schedules", "presence_sessions", "recognition_evidence",
            "enrollment_operations", "attendance_decisions", "liveness_challenges",
            "platform_audit_log", "incidents", "incident_events",
            "incident_review_actions", "incident_alerts", "incident_evidence",
            "platform_users", "platform_sessions", "command_idempotency",
            "platform_outbox", "edge_sync_devices", "edge_sync_batches",
            "edge_sync_events", "attendance_corrections", "cybersecurity_events",
            "cybersecurity_incidents", "cybersecurity_incident_events",
            "cybersecurity_incident_alerts", "cybersecurity_reviews",
            "schema_migration_runs",
        )
    else:
        return None
    payload: dict[str, Any] = {"tables": []}
    for table_name in tables_to_fingerprint:
        if not _table_exists(conn, table_name):
            return None
        table_sql = conn.execute(
            "select sql from sqlite_master where type='table' and name=?", (table_name,)
        ).fetchone()
        columns = [tuple(row[:5]) for row in conn.execute(f"pragma table_info({table_name})").fetchall()]
        indexes: list[tuple[object, ...]] = []
        for row in conn.execute(f"pragma index_list({table_name})").fetchall():
            index_name = str(row[1])
            index_sql = conn.execute(
                "select sql from sqlite_master where type='index' and name=?", (index_name,)
            ).fetchone()
            escaped = index_name.replace('"', '""')
            index_columns = [
                tuple(item[:3]) for item in conn.execute(f'pragma index_info("{escaped}")').fetchall()
            ]
            indexes.append((index_name, int(row[2] or 0), _normalise_sql(index_sql[0] if index_sql else ""), index_columns))
        payload["tables"].append({
            "name": table_name,
            "table": _normalise_sql(table_sql[0] if table_sql else ""),
            "columns": columns,
            "indexes": sorted(indexes, key=lambda item: str(item[0])),
        })
    trigger_rows = conn.execute(
        "select name, tbl_name, sql from sqlite_master where type='trigger' order by name"
    ).fetchall()
    if schema_name == "outbox":
        # The outbox contract must not drift merely because an unrelated
        # platform table gained a protection trigger.
        trigger_rows = [row for row in trigger_rows if str(row[1]) in tables_to_fingerprint]
    payload["triggers"] = [
        (str(row[0]), str(row[1]), _normalise_sql(row[2])) for row in trigger_rows
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _canonical_audit_payload(row: dict[str, Any], previous_hash: str) -> bytes:
    # Version 1 rows predate deployment scope columns. Version 2 rows include
    # scope but predate the explicit version marker. Version 3 is the current
    # contract and binds the marker as well. Preserve both historical
    # contracts instead of rewriting append-only evidence during migration.
    try:
        hash_version = int(row.get("hash_version") or 1)
    except (TypeError, ValueError):
        hash_version = 1
    payload = {
        key: value for key, value in row.items()
        if key not in {"record_hash", "prev_hash"}
    }
    if hash_version < 2:
        payload = {
            key: value for key, value in payload.items()
            if key not in {"hash_version", "organization_id", "site_id", "device_id"}
        }
    elif hash_version == 2:
        payload.pop("hash_version", None)
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


def _ensure_migration_history(conn: sqlite3.Connection) -> None:
    """Create the append-only record of applied adapter contracts."""
    conn.execute(
        """
        create table if not exists schema_migration_history (
            id integer primary key autoincrement,
            schema_name text not null,
            version integer not null,
            checksum text not null,
            status text not null default 'applied',
            applied_at text not null,
            process_id integer,
            host_name text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_schema_migration_history_time "
        "on schema_migration_history(schema_name, applied_at)"
    )
    conn.executescript(
        """
        create trigger if not exists trg_schema_migration_history_update
        before update on schema_migration_history begin
            select raise(abort, 'append-only migration history');
        end;
        create trigger if not exists trg_schema_migration_history_delete
        before delete on schema_migration_history begin
            select raise(abort, 'append-only migration history');
        end;
        """
    )
    conn.execute(
        """
        create table if not exists schema_migration_runs (
            id integer primary key autoincrement,
            run_id text not null,
            schema_name text not null,
            target_version integer not null,
            status text not null check (status in ('started', 'applied', 'failed')),
            error_code text,
            occurred_at text not null,
            process_id integer,
            host_name text
        )
        """
    )
    conn.execute(
        "create index if not exists idx_schema_migration_runs_time "
        "on schema_migration_runs(schema_name, occurred_at, id)"
    )
    conn.executescript(
        """
        create trigger if not exists trg_schema_migration_runs_update
        before update on schema_migration_runs begin
            select raise(abort, 'append-only migration run');
        end;
        create trigger if not exists trg_schema_migration_runs_delete
        before delete on schema_migration_runs begin
            select raise(abort, 'append-only migration run');
        end;
        """
    )


def start_schema_migration_run(conn: sqlite3.Connection, schema_name: str, target_version: int) -> str:
    """Persist a migration start marker before schema DDL is attempted."""
    _ensure_migration_history(conn)
    run_id = uuid.uuid4().hex
    conn.execute(
        """
        insert into schema_migration_runs
            (run_id, schema_name, target_version, status, occurred_at, process_id, host_name)
        values (?, ?, ?, 'started', ?, ?, ?)
        """,
        (run_id, str(schema_name), int(target_version), _utc_now(), os.getpid(), socket.gethostname()[:255]),
    )
    return run_id


def finish_schema_migration_run(
    conn: sqlite3.Connection,
    run_id: str,
    schema_name: str,
    target_version: int,
    status: str,
    error_code: str | None = None,
) -> None:
    """Append the terminal state of a previously started migration run."""
    if status not in {"applied", "failed"}:
        raise ValueError("migration run status must be applied or failed")
    _ensure_migration_history(conn)
    conn.execute(
        """
        insert into schema_migration_runs
            (run_id, schema_name, target_version, status, error_code, occurred_at, process_id, host_name)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (str(run_id), str(schema_name), int(target_version), status, str(error_code or "")[:300] or None,
         _utc_now(), os.getpid(), socket.gethostname()[:255]),
    )


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent storage migrations without dropping existing data."""
    _ensure_migration_history(conn)
    conn.execute(
        """
        create table if not exists schema_migrations (
            version integer primary key,
            description text not null,
            checksum text,
            applied_at text not null
        )
        """
    )
    migration_columns = _columns(conn, "schema_migrations")
    if "checksum" not in migration_columns:
        conn.execute("alter table schema_migrations add column checksum text")
    for version, checksum in EXPECTED_MIGRATION_CHECKSUMS.items():
        conn.execute(
            "update schema_migrations set checksum=? where version=? and (checksum is null or checksum='')",
            (checksum, version),
        )
    conn.execute(
        """
        create table if not exists schema_migration_state (
            schema_name text primary key,
            version integer not null,
            description text not null,
            checksum text not null,
            schema_fingerprint text,
            applied_at text not null
        )
        """
    )
    if "schema_fingerprint" not in _columns(conn, "schema_migration_state"):
        conn.execute("alter table schema_migration_state add column schema_fingerprint text")

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
        if "hash_version" not in columns:
            # Existing rows are deliberately treated as version 1. New rows
            # use version 2 and bind deployment scope into the hash payload.
            conn.execute(
                f"alter table {table} add column hash_version integer not null default 1"
            )
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
            created_at text not null,
            signature text
        )
        """
    )
    if "signature" not in _columns(conn, "audit_checkpoints"):
        conn.execute("alter table audit_checkpoints add column signature text")

    if 1 not in applied:
        conn.execute(
            "insert into schema_migrations(version, description, checksum, applied_at) values (?, ?, ?, ?)",
            (1, "tamper-evident audit chains and shared migration ledger", EXPECTED_MIGRATION_CHECKSUMS[1], _utc_now()),
        )


def record_schema_state(
    conn: sqlite3.Connection,
    schema_name: str,
    version: int,
    description: str,
    checksum: str,
) -> None:
    """Record the versioned contract applied by one database adapter.

    ``schema_migrations`` remains the compatibility ledger for the original
    audit-chain migration. This separate state table prevents the edge and
    API adapters from pretending one global integer describes two owners,
    while keeping startup state inspectable and versioned.
    """
    _ensure_migration_history(conn)
    conn.execute(
        """
        create table if not exists schema_migration_state (
            schema_name text primary key,
            version integer not null,
            description text not null,
            checksum text not null,
            schema_fingerprint text,
            applied_at text not null
        )
        """
    )
    if "schema_fingerprint" not in _columns(conn, "schema_migration_state"):
        conn.execute("alter table schema_migration_state add column schema_fingerprint text")

    fingerprint = schema_fingerprint(conn, str(schema_name))

    conn.execute(
        """
        insert into schema_migration_state(schema_name, version, description, checksum, schema_fingerprint, applied_at)
        values (?, ?, ?, ?, ?, ?)
        on conflict(schema_name) do update set
          version=excluded.version,
          description=excluded.description,
          checksum=excluded.checksum,
          schema_fingerprint=case
            when schema_migration_state.version < excluded.version then excluded.schema_fingerprint
            else coalesce(schema_migration_state.schema_fingerprint, excluded.schema_fingerprint)
          end,
          applied_at=excluded.applied_at
        """,
        (str(schema_name), int(version), str(description), str(checksum), fingerprint, _utc_now()),
    )
    conn.execute(
        """
        insert into schema_migration_history
            (schema_name, version, checksum, status, applied_at, process_id, host_name)
        values (?, ?, ?, 'applied', ?, ?, ?)
        """,
        (
            str(schema_name),
            int(version),
            str(checksum),
            _utc_now(),
            os.getpid(),
            socket.gethostname()[:255],
        ),
    )


def refresh_schema_state_fingerprint(conn: sqlite3.Connection, schema_name: str) -> None:
    """Refresh one fingerprint after an explicitly coordinated cross-adapter DDL step.

    Normal health checks must keep detecting unexpected drift. This narrow
    helper is reserved for a known migration boundary where another adapter
    has just added contract objects, such as the platform scope triggers on
    the shared outbox table.
    """
    if not _table_exists(conn, "schema_migration_state"):
        return
    fingerprint = schema_fingerprint(conn, str(schema_name))
    conn.execute(
        "update schema_migration_state set schema_fingerprint=? where schema_name=?",
        (fingerprint, str(schema_name)),
    )


def schema_state_issues(
    conn: sqlite3.Connection,
    schema_names: tuple[str, ...] = ("outbox",),
) -> list[str]:
    """Verify recorded migration state against the known application contract."""
    exists = conn.execute(
        "select 1 from sqlite_master where type='table' and name='schema_migration_state'"
    ).fetchone()
    if not exists:
        return ["missing table: schema_migration_state"]
    issues: list[str] = []
    state_columns = _columns(conn, "schema_migration_state")
    select_fingerprint = ", schema_fingerprint" if "schema_fingerprint" in state_columns else ""
    rows: dict[str, dict[str, Any]] = {}
    for raw in conn.execute(
        f"select schema_name, version, checksum{select_fingerprint} from schema_migration_state"
    ).fetchall():
        if isinstance(raw, sqlite3.Row):
            row = dict(raw)
        else:
            row = {"schema_name": raw[0], "version": raw[1], "checksum": raw[2]}
            if select_fingerprint:
                row["schema_fingerprint"] = raw[3]
        rows[str(row["schema_name"])] = row
    for name in schema_names:
        expected = EXPECTED_SCHEMA_STATES.get(name)
        if expected is None:
            issues.append(f"unknown expected schema state: {name}")
            continue
        row = rows.get(name)
        if not row:
            issues.append(f"missing schema state: {name}")
            continue
        version, checksum = expected
        if int(row.get("version") or 0) != version:
            issues.append(f"schema version mismatch: {name}")
        if not hmac.compare_digest(str(row.get("checksum") or ""), checksum):
            issues.append(f"schema checksum mismatch: {name}")
        expected_fingerprint = schema_fingerprint(conn, name)
        recorded_fingerprint = str(row.get("schema_fingerprint") or "")
        if expected_fingerprint and not recorded_fingerprint:
            issues.append(f"missing schema fingerprint: {name}")
        elif expected_fingerprint and not hmac.compare_digest(recorded_fingerprint, expected_fingerprint):
            issues.append(f"schema fingerprint mismatch: {name}")
    return issues


def migration_issues(conn: sqlite3.Connection) -> list[str]:
    """Verify the shared migration ledger itself has not been altered."""
    if not _table_exists(conn, "schema_migrations"):
        return ["missing table: schema_migrations"]
    columns = _columns(conn, "schema_migrations")
    if "checksum" not in columns:
        return ["missing migration checksum column"]
    rows = {
        int(row[0]): str(row[1] or "")
        for row in conn.execute("select version, checksum from schema_migrations").fetchall()
    }
    issues: list[str] = []
    for version, expected in EXPECTED_MIGRATION_CHECKSUMS.items():
        actual = rows.get(version)
        if not actual:
            issues.append(f"missing migration checksum: {version}")
        elif not hmac.compare_digest(actual, expected):
            issues.append(f"migration checksum mismatch: {version}")
    return issues


def _audit_checkpoint_bytes(table: str, record_id: int, record_hash: str, created_at: str) -> bytes:
    return json.dumps(
        {
            "table_name": str(table),
            "last_record_id": int(record_id),
            "last_record_hash": str(record_hash),
            "created_at": str(created_at),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _audit_signing_key() -> str:
    return os.getenv("OPTIVOX_AUDIT_SIGNING_KEY", "").strip()


def create_audit_checkpoint(conn: sqlite3.Connection, table: str) -> dict[str, Any] | None:
    """Sign the current audit-chain tip inside the caller's transaction."""
    if not _table_exists(conn, "audit_checkpoints") or not _table_exists(conn, table):
        return None
    columns = _columns(conn, "audit_checkpoints")
    if "signature" not in columns:
        conn.execute("alter table audit_checkpoints add column signature text")
    tip = conn.execute(
        f"select id, record_hash from {table} order by id desc limit 1"
    ).fetchone()
    if not tip:
        return None
    created_at = _utc_now()
    key = _audit_signing_key().encode("utf-8")
    signature = hmac.new(
        key,
        _audit_checkpoint_bytes(table, int(tip[0]), str(tip[1] or ""), created_at),
        hashlib.sha256,
    ).hexdigest() if key else None
    checkpoint_id = conn.execute(
        """insert into audit_checkpoints
           (table_name, last_record_id, last_record_hash, created_at, signature)
           values (?, ?, ?, ?, ?)""",
        (table, int(tip[0]), str(tip[1] or ""), created_at, signature),
    ).lastrowid
    return {
        "id": int(checkpoint_id),
        "table_name": table,
        "last_record_id": int(tip[0]),
        "last_record_hash": str(tip[1] or ""),
        "created_at": created_at,
        "signed": bool(signature),
    }


def consistency_issues(conn: sqlite3.Connection) -> list[str]:
    """Detect semantic inconsistencies shared by edge and platform adapters."""
    issues: list[str] = []
    tables = {
        row[0] for row in conn.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()
    }
    checks = (
        (
            "duplicate active presence sessions",
            {"presence_sessions"},
            """select count(*) from (
                 select organization_id, site_id, entity_id, camera_id
                 from presence_sessions
                 where status in ('active', 'occluded')
                 group by organization_id, site_id, entity_id, camera_id
                 having count(*) > 1
               )""",
        ),
        (
            "incidents without linked observations",
            {"incidents", "incident_events"},
            """select count(*) from incidents i
               where not exists (select 1 from incident_events x where x.incident_id=i.id)
                 and i.status not in ('dismissed', 'resolved')""",
        ),
        (
            "cyber incidents without linked events",
            {"cybersecurity_incidents", "cybersecurity_incident_events"},
            """select count(*) from cybersecurity_incidents i
               where not exists (
                 select 1 from cybersecurity_incident_events x where x.incident_id=i.id
               )
                 and i.status not in ('dismissed', 'resolved')""",
        ),
        (
            "attendance decisions missing an evidence reference",
            {"attendance_decisions"},
            """select count(*) from attendance_decisions d
               where lower(coalesce(d.decision, '')) in ('clock_in', 'eligible', 'accepted', 'confirmed')
                 and d.person_id is not null
                 and d.recognition_evidence_id is null""",
        ),
        (
            "attendance clock-out precedes clock-in",
            {"attendance"},
            """select count(*) from attendance
               where clock_in is not null and clock_out is not null
                 and julianday(clock_out) < julianday(clock_in)""",
        ),
        (
            "attendance contains negative durations",
            {"attendance"},
            """select count(*) from attendance
               where coalesce(work_minutes, 0) < 0
                  or coalesce(late_minutes, 0) < 0
                  or coalesce(early_departure_minutes, 0) < 0""",
        ),
        (
            "presence session ended before it started",
            {"presence_sessions"},
            """select count(*) from presence_sessions
               where ended_at is not null and julianday(ended_at) < julianday(started_at)""",
        ),
        (
            "attendance contains an unsupported status",
            {"attendance"},
            """select count(*) from attendance
               where lower(trim(coalesce(attendance_status, ''))) not in
                 ('recorded', 'manual', 'present', 'late', 'completed', 'early_departure', 'corrected', 'excused', 'absent')""",
        ),
        (
            "incident contains an unsupported status",
            {"incidents"},
            """select count(*) from incidents
               where lower(trim(coalesce(status, ''))) not in
                 ('open', 'acknowledged', 'assigned', 'escalated', 'confirmed', 'dismissed', 'resolved')""",
        ),
        (
            "cyber incident contains an unsupported status",
            {"cybersecurity_incidents"},
            """select count(*) from cybersecurity_incidents
               where lower(trim(coalesce(status, ''))) not in
                 ('open', 'acknowledged', 'assigned', 'escalated', 'confirmed', 'dismissed', 'resolved')""",
        ),
        (
            "attendance references a missing person",
            {"attendance", "people"},
            """select count(*) from attendance a
               left join people p on p.id=a.person_id
               where a.person_id is not null and p.id is null""",
        ),
        (
            "presence session references a missing person",
            {"presence_sessions", "people"},
            """select count(*) from presence_sessions s
               left join people p on p.id=s.person_id
               where s.person_id is not null and p.id is null""",
        ),
        (
            "recognition evidence references a missing person",
            {"recognition_evidence", "people"},
            """select count(*) from recognition_evidence e
               left join people p on p.id=e.person_id
               where e.person_id is not null and p.id is null""",
        ),
    )
    for label, required_tables, query in checks:
        if not required_tables.issubset(tables):
            continue
        count = int(conn.execute(query).fetchone()[0] or 0)
        if count:
            issues.append(f"{label}: {count}")

    # SQLite does not provide PostgreSQL-style row-level security. The local
    # adapters therefore enforce scope in service queries, while this health
    # check catches records that could bypass that boundary through a direct
    # import, legacy script, or damaged migration.
    scoped_tables = (
        "people", "events", "attendance", "absence_records",
        "attendance_schedules", "presence_sessions", "recognition_evidence",
        "enrollment_operations", "attendance_decisions", "liveness_challenges",
        "incidents", "incident_events", "incident_review_actions",
        "incident_alerts", "incident_evidence", "attendance_corrections",
        "alert_log", "platform_audit_log", "platform_outbox",
        "cybersecurity_events", "cybersecurity_incidents",
        "cybersecurity_incident_events", "cybersecurity_incident_alerts",
        "cybersecurity_reviews",
        "edge_sync_devices", "edge_sync_batches", "edge_sync_events",
    )
    for table in scoped_tables:
        if table not in tables:
            continue
        columns = _columns(conn, table)
        if not {"organization_id", "site_id", "device_id"}.issubset(columns):
            continue
        count = int(conn.execute(
            f"""select count(*) from {table}
                where nullif(trim(coalesce(organization_id, '')), '') is null
                   or nullif(trim(coalesce(site_id, '')), '') is null
                   or nullif(trim(coalesce(device_id, '')), '') is null"""
        ).fetchone()[0] or 0)
        if count:
            issues.append(f"scoped rows missing deployment identity in {table}: {count}")

    cross_scope_checks = (
        (
            "attendance references a person from another scope",
            {"attendance", "people"},
            """select count(*) from attendance a join people p on p.id=a.person_id
               where a.person_id is not null
                 and a.organization_id is not null and a.site_id is not null
                 and p.organization_id is not null and p.site_id is not null
                 and (a.organization_id<>p.organization_id or a.site_id<>p.site_id)""",
        ),
        (
            "recognition evidence references a person from another scope",
            {"recognition_evidence", "people"},
            """select count(*) from recognition_evidence e join people p on p.id=e.person_id
               where e.person_id is not null
                 and e.organization_id is not null and e.site_id is not null
                 and p.organization_id is not null and p.site_id is not null
                 and (e.organization_id<>p.organization_id or e.site_id<>p.site_id)""",
        ),
        (
            "attendance decision references a person from another scope",
            {"attendance_decisions", "people"},
            """select count(*) from attendance_decisions d join people p on p.id=d.person_id
               where d.person_id is not null
                 and d.organization_id is not null and d.site_id is not null
                 and p.organization_id is not null and p.site_id is not null
               and (d.organization_id<>p.organization_id or d.site_id<>p.site_id)""",
        ),
        (
            "incident event link crosses deployment scope",
            {"incident_events", "incidents", "events"},
            """select count(*) from incident_events x
               join incidents i on i.id=x.incident_id
               join events e on e.id=x.event_id
               where i.organization_id is not null and i.site_id is not null
                 and e.organization_id is not null and e.site_id is not null
                 and (i.organization_id<>e.organization_id or i.site_id<>e.site_id)""",
        ),
        (
            "incident evidence link crosses deployment scope",
            {"incident_evidence", "incidents", "events"},
            """select count(*) from incident_evidence x
               join incidents i on i.id=x.incident_id
               join events e on e.id=x.event_id
               where i.organization_id is not null and i.site_id is not null
                 and e.organization_id is not null and e.site_id is not null
                 and (i.organization_id<>e.organization_id or i.site_id<>e.site_id)""",
        ),
        (
            "absence references a person from another scope",
            {"absence_records", "people"},
            """select count(*) from absence_records a join people p on p.id=a.person_id
               where a.person_id is not null
                 and (a.organization_id<>p.organization_id or a.site_id<>p.site_id)""",
        ),
        (
            "presence session references a person from another scope",
            {"presence_sessions", "people"},
            """select count(*) from presence_sessions s join people p on p.id=s.person_id
               where s.person_id is not null
                 and (s.organization_id<>p.organization_id or s.site_id<>p.site_id)""",
        ),
        (
            "recognition evidence link crosses presence scope",
            {"recognition_evidence", "presence_sessions"},
            """select count(*) from recognition_evidence e
               join presence_sessions s on s.id=e.presence_session_id
               where e.presence_session_id is not null
                 and (e.organization_id<>s.organization_id or e.site_id<>s.site_id)""",
        ),
        (
            "attendance decision link crosses evidence scope",
            {"attendance_decisions", "recognition_evidence"},
            """select count(*) from attendance_decisions d
               join recognition_evidence e on e.id=d.recognition_evidence_id
               where d.recognition_evidence_id is not null
                 and (d.organization_id<>e.organization_id or d.site_id<>e.site_id)""",
        ),
        (
            "incident event link crosses event scope",
            {"incident_events", "incidents", "events"},
            """select count(*) from incident_events x
               join incidents i on i.id=x.incident_id
               join events e on e.id=x.event_id
               where i.organization_id<>e.organization_id or i.site_id<>e.site_id""",
        ),
        (
            "incident review crosses incident scope",
            {"incident_review_actions", "incidents"},
            """select count(*) from incident_review_actions r join incidents i on i.id=r.incident_id
               where r.organization_id<>i.organization_id or r.site_id<>i.site_id""",
        ),
        (
            "incident alert crosses incident scope",
            {"incident_alerts", "incidents"},
            """select count(*) from incident_alerts a join incidents i on i.id=a.incident_id
               where a.organization_id<>i.organization_id or a.site_id<>i.site_id""",
        ),
    )
    for label, required_tables, query in cross_scope_checks:
        if not required_tables.issubset(tables):
            continue
        count = int(conn.execute(query).fetchone()[0] or 0)
        if count:
            issues.append(f"{label}: {count}")
    return issues


def schema_object_issues(conn: sqlite3.Connection) -> list[str]:
    """Verify the indexes and triggers that enforce the database contract."""
    tables = {
        str(row[0]) for row in conn.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()
    }
    required_indexes = {
        "presence_sessions": ("idx_presence_one_open_per_entity",),
        "events": ("idx_events_event_uid",),
        "platform_outbox": ("idx_outbox_scope_status",),
        "platform_audit_log": ("idx_audit_scope_time",),
    }
    issues: list[str] = []
    for table, names in required_indexes.items():
        if table not in tables:
            continue
        actual = {
            str(row[1]) for row in conn.execute(f'pragma index_list("{table}")').fetchall()
        }
        for name in names:
            if name not in actual:
                issues.append(f"missing required index: {name}")

    required_triggers = {
        "trg_scope_events_session": "events",
        "trg_scope_attendance_person": "attendance",
        "trg_scope_absence_person": "absence_records",
        "trg_scope_presence_person": "presence_sessions",
        "trg_scope_evidence_links": "recognition_evidence",
        "trg_scope_attendance_decision_links": "attendance_decisions",
        "trg_scope_incident_event": "incident_events",
        "trg_scope_incident_evidence": "incident_evidence",
        "trg_scope_incident_review": "incident_review_actions",
        "trg_scope_incident_alert": "incident_alerts",
        "trg_scope_cyber_event_incident": "cybersecurity_events",
        "trg_scope_cyber_incident_event": "cybersecurity_incident_events",
        "trg_scope_cyber_alert": "cybersecurity_incident_alerts",
        "trg_scope_cyber_review": "cybersecurity_reviews",
        "trg_platformauditlog_append_only_update": "platform_audit_log",
        "trg_platformauditlog_append_only_delete": "platform_audit_log",
        "trg_auditlog_append_only_update": "audit_log",
        "trg_auditlog_append_only_delete": "audit_log",
        "trg_auditcheckpoints_append_only_update": "audit_checkpoints",
        "trg_auditcheckpoints_append_only_delete": "audit_checkpoints",
        "trg_schema_migration_history_update": "schema_migration_history",
        "trg_schema_migration_history_delete": "schema_migration_history",
        "trg_schema_migration_runs_update": "schema_migration_runs",
        "trg_schema_migration_runs_delete": "schema_migration_runs",
        "trg_attendancecorrections_append_only_update": "attendance_corrections",
        "trg_attendancecorrections_append_only_delete": "attendance_corrections",
        "trg_incidentreviewactions_append_only_update": "incident_review_actions",
        "trg_incidentreviewactions_append_only_delete": "incident_review_actions",
        "trg_incidentalerts_append_only_update": "incident_alerts",
        "trg_incidentalerts_append_only_delete": "incident_alerts",
        "trg_policy_snapshots_immutable_update": "policy_snapshots",
        "trg_policy_snapshots_immutable_delete": "policy_snapshots",
    }
    for table in (
        "people", "events", "attendance", "absence_records",
        "attendance_schedules", "presence_sessions", "recognition_evidence",
        "enrollment_operations", "attendance_decisions", "liveness_challenges",
        "incidents", "incident_events", "incident_review_actions",
        "incident_alerts", "incident_evidence", "alert_log", "platform_audit_log",
        "attendance_corrections", "platform_outbox", "cybersecurity_events",
        "cybersecurity_incidents", "cybersecurity_incident_events",
        "cybersecurity_incident_alerts", "cybersecurity_reviews",
        "edge_sync_devices", "edge_sync_batches", "edge_sync_events",
    ):
        required_triggers[f"trg_scope_immutable_{table}"] = table
    for table, column in (
        ("attendance", "attendance_status"),
        ("presence_sessions", "identity_state"),
        ("presence_sessions", "liveness_status"),
        ("recognition_evidence", "identity_state"),
        ("recognition_evidence", "liveness_status"),
        ("attendance_decisions", "identity_state"),
        ("attendance_decisions", "liveness_status"),
        ("liveness_challenges", "liveness_status"),
        ("liveness_challenges", "challenge_state"),
        ("incidents", "status"),
        ("cybersecurity_incidents", "status"),
    ):
        required_triggers[f"trg_domain_{table}_{column}_insert"] = table
        required_triggers[f"trg_domain_{table}_{column}_update"] = table
    for table, suffix in (
        ("attendance", "time_insert"),
        ("attendance", "time_update"),
        ("presence_sessions", "time_insert"),
        ("presence_sessions", "time_update"),
    ):
        required_triggers[f"trg_domain_{table}_{suffix}"] = table
    for operation in ("insert", "update"):
        required_triggers[f"trg_domain_attendance_nonnegative_{operation}"] = "attendance"
    actual_triggers = {
        str(row[0]) for row in conn.execute(
            "select name from sqlite_master where type='trigger'"
        ).fetchall()
    }
    for name, table in required_triggers.items():
        if table in tables and name not in actual_triggers:
            issues.append(f"missing required trigger: {name}")
    return issues


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
    if "hash_version" in columns and "hash_version" not in row_values:
        row_values["hash_version"] = 3
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
    hash_versions: dict[str, int] = {}
    version_mismatches = 0
    for raw in rows:
        row = dict(raw) if isinstance(raw, sqlite3.Row) else dict(zip(
            [item[1] for item in conn.execute(f"pragma table_info({table})").fetchall()], raw
        ))
        if str(row.get("prev_hash") or "") != previous_hash:
            return {"ok": False, "table": table, "reason": "previous_hash_mismatch", "record_id": row.get("id"), "checked": row.get("id", 0)}
        try:
            declared_version = int(row.get("hash_version") or 1)
        except (TypeError, ValueError):
            declared_version = 1
        candidate_versions = [declared_version]
        # Databases upgraded after scope columns were added received a
        # default version value but their historical hash did not include the
        # new marker. Accept that immutable intermediate contract once.
        if declared_version == 1:
            candidate_versions.append(2)
        matched_version = None
        for candidate_version in candidate_versions:
            candidate = dict(row)
            candidate["hash_version"] = candidate_version
            expected = audit_record_hash(candidate, previous_hash)
            if row.get("record_hash") and hmac_compare(str(row["record_hash"]), expected):
                matched_version = candidate_version
                break
        if matched_version is None:
            return {"ok": False, "table": table, "reason": "record_hash_mismatch", "record_id": row.get("id"), "checked": row.get("id", 0)}
        if matched_version != declared_version:
            version_mismatches += 1
        hash_versions[str(matched_version)] = hash_versions.get(str(matched_version), 0) + 1
        previous_hash = str(row["record_hash"])
    checkpoint_status = "not_configured"
    checkpoint_table_exists = _table_exists(conn, "audit_checkpoints")
    if checkpoint_table_exists:
        checkpoints = conn.execute(
            "select * from audit_checkpoints where table_name=? order by id desc limit 1",
            (table,),
        ).fetchall()
        key = _audit_signing_key().encode("utf-8")
        if checkpoints:
            checkpoint_status = "unsigned" if not key else "valid"
            for raw in checkpoints:
                checkpoint = dict(raw) if isinstance(raw, sqlite3.Row) else dict(zip(
                    [item[1] for item in conn.execute("pragma table_info(audit_checkpoints)").fetchall()], raw
                ))
                checkpoint_row = conn.execute(
                    f"select record_hash from {table} where id=?",
                    (checkpoint.get("last_record_id"),),
                ).fetchone()
                if not checkpoint_row or str(checkpoint_row[0] or "") != str(checkpoint.get("last_record_hash") or ""):
                    return {
                        "ok": False,
                        "table": table,
                        "reason": "checkpoint_tip_mismatch",
                        "record_id": checkpoint.get("last_record_id"),
                        "checked": len(rows),
                    }
                if key:
                    signature = str(checkpoint.get("signature") or "")
                    expected_signature = hmac.new(
                        key,
                        _audit_checkpoint_bytes(
                            table,
                            int(checkpoint.get("last_record_id") or 0),
                            str(checkpoint.get("last_record_hash") or ""),
                            str(checkpoint.get("created_at") or ""),
                        ),
                        hashlib.sha256,
                    ).hexdigest()
                    if not signature or not hmac.compare_digest(signature, expected_signature):
                        return {
                            "ok": False,
                            "table": table,
                            "reason": "checkpoint_signature_mismatch",
                            "record_id": checkpoint.get("last_record_id"),
                            "checked": len(rows),
                        }
        elif key and os.getenv("OPTIVOX_RUNTIME_MODE", "development").strip().lower() in {"pilot", "production"}:
            checkpoint_status = "missing"
            return {"ok": False, "table": table, "reason": "checkpoint_missing", "checked": len(rows)}
    legacy_count = hash_versions.get("1", 0)
    result = {
        "ok": True,
        "table": table,
        "checked": len(rows),
        "last_hash": previous_hash,
        "checkpoint_status": checkpoint_status,
        "hash_versions": hash_versions,
        "legacy_unscoped_records": legacy_count,
        "version_mismatches": version_mismatches,
    }
    if legacy_count:
        result["integrity_level"] = "legacy_compatible"
        result["warnings"] = ["historical audit rows use the pre-scope hash contract"]
    elif hash_versions.get("2", 0):
        result["integrity_level"] = "scope_bound_legacy_version"
        result["warnings"] = [
            "some historical audit rows are scope-bound but predate the explicit hash-version marker"
        ]
    else:
        result["integrity_level"] = "scope_bound"
    return result


def hmac_compare(left: str, right: str) -> bool:
    # Kept local to avoid making callers depend on the authentication module.
    import hmac
    return hmac.compare_digest(left, right)
