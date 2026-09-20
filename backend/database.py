from __future__ import annotations

import sqlite3
import os
import hashlib
import importlib
import shutil
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DATABASE_MAX_BYTES,
    DATABASE_ENCRYPTION_DRIVER,
    DATABASE_ENCRYPTION_ENABLED,
    DATABASE_ENCRYPTION_KEY,
    DATABASE_MIN_FREE_BYTES,
    DATABASE_PATH,
    DATABASE_WAL_MAX_BYTES,
    DATABASE_WARN_BYTES,
    SQLITE_SECURE_DELETE,
    SQLITE_SYNCHRONOUS,
)
from schema_migrations import consistency_issues, migration_issues, schema_object_issues, schema_state_issues, verify_audit_chain


class DatabaseError(RuntimeError):
    pass


_MIGRATION_LOCKS: dict[str, threading.RLock] = {}
_MIGRATION_LOCKS_GUARD = threading.Lock()
_MIGRATION_LOCK_HELD = threading.local()
_METRICS_LOCK = threading.Lock()
_METRICS = {
    "connections_opened": 0,
    "connections_closed": 0,
    "connection_errors": 0,
    "transactions_started": 0,
    "transactions_committed": 0,
    "transactions_rolled_back": 0,
    "operation_errors": 0,
    "lock_errors": 0,
}
_CONNECTION_LATENCIES_MS: deque[float] = deque(maxlen=256)
_TRANSACTION_LATENCIES_MS: deque[float] = deque(maxlen=256)


def _metric_increment(name: str) -> None:
    with _METRICS_LOCK:
        _METRICS[name] = int(_METRICS.get(name, 0)) + 1


def _metric_latency(target: deque[float], milliseconds: float) -> None:
    with _METRICS_LOCK:
        target.append(max(0.0, float(milliseconds)))


def _is_lock_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(float(ordered[index]), 3)


def database_metrics() -> dict[str, Any]:
    """Return bounded database operational metrics without query payloads."""
    with _METRICS_LOCK:
        counters = dict(_METRICS)
        connection_latencies = list(_CONNECTION_LATENCIES_MS)
        transaction_latencies = list(_TRANSACTION_LATENCIES_MS)
    return {
        **counters,
        "samples": {
            "connection_latency_ms": len(connection_latencies),
            "transaction_latency_ms": len(transaction_latencies),
        },
        "latency_ms": {
            "connection_p50": _percentile(connection_latencies, 0.50),
            "connection_p95": _percentile(connection_latencies, 0.95),
            "transaction_p50": _percentile(transaction_latencies, 0.50),
            "transaction_p95": _percentile(transaction_latencies, 0.95),
        },
    }


def reset_database_metrics() -> None:
    """Reset process-local metrics for a controlled benchmark or test."""
    with _METRICS_LOCK:
        for key in _METRICS:
            _METRICS[key] = 0
        _CONNECTION_LATENCIES_MS.clear()
        _TRANSACTION_LATENCIES_MS.clear()


def _load_sqlcipher_driver():
    """Load an explicitly requested SQLCipher-compatible DB-API module."""
    for module_name in ("pysqlcipher3.dbapi2", "sqlcipher3.dbapi2"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def database_encryption_status() -> dict[str, Any]:
    """Return capability metadata without exposing the database key."""
    if not DATABASE_ENCRYPTION_ENABLED:
        return {"status": "NOT_CONFIGURED", "enabled": False, "driver": None}
    driver = _load_sqlcipher_driver()
    return {
        "status": "READY" if driver is not None and len(DATABASE_ENCRYPTION_KEY) >= 32 else "UNAVAILABLE",
        "enabled": True,
        "driver": DATABASE_ENCRYPTION_DRIVER,
        "key_configured": bool(DATABASE_ENCRYPTION_KEY),
    }


def connect_database(path=None, *, timeout: float = 10.0):
    """Open the configured database with optional SQLCipher protection."""
    resolved = path or DATABASE_PATH
    driver = _load_sqlcipher_driver() if DATABASE_ENCRYPTION_ENABLED else sqlite3
    if DATABASE_ENCRYPTION_ENABLED:
        if driver is None:
            raise DatabaseError("SQLCipher-compatible database driver is not installed.")
        if len(DATABASE_ENCRYPTION_KEY) < 32:
            raise DatabaseError("OPTIVOX_DATABASE_ENCRYPTION_KEY is missing or too short.")
    con = driver.connect(resolved, timeout=timeout, check_same_thread=False)
    if DATABASE_ENCRYPTION_ENABLED:
        # SQLCipher accepts a hexadecimal key literal; hashing the configured
        # secret avoids interpolation of operator-provided characters.
        key_hex = hashlib.sha256(DATABASE_ENCRYPTION_KEY.encode("utf-8")).hexdigest()
        con.execute(f'PRAGMA key = "x\'{key_hex}\'"')
        try:
            con.execute("select count(*) from sqlite_master").fetchone()
        except Exception as exc:
            con.close()
            raise DatabaseError("Encrypted database authentication failed.") from exc
    con.row_factory = getattr(driver, "Row", sqlite3.Row)
    return con


def _migration_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve()).casefold()
    with _MIGRATION_LOCKS_GUARD:
        return _MIGRATION_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def database_migration_lock(path=None, *, timeout_seconds: float = 30.0):
    """Serialize schema changes across threads and independent processes.

    SQLite's busy timeout protects individual statements, but it does not make
    a multi-statement additive migration an atomic application-level action.
    A tiny sidecar lock file gives the edge runtime and FastAPI process the
    same migration boundary without putting lock state inside a schema that is
    itself being created. The lock is re-entrant for nested initialization in
    one runtime process.
    """
    resolved = Path(path or DATABASE_PATH).expanduser().resolve()
    thread_lock = _migration_lock_for(resolved)
    with thread_lock:
        held = getattr(_MIGRATION_LOCK_HELD, "paths", set())
        key = str(resolved).casefold()
        if key in held:
            yield
            return

        lock_path = Path(f"{resolved}.schema.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                deadline = time.monotonic() + max(0.1, float(timeout_seconds))
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise DatabaseError("Timed out waiting for the database migration lock.")
                        time.sleep(0.05)
            else:
                import fcntl
                deadline = time.monotonic() + max(0.1, float(timeout_seconds))
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise DatabaseError("Timed out waiting for the database migration lock.")
                        time.sleep(0.05)
            held = set(held)
            held.add(key)
            _MIGRATION_LOCK_HELD.paths = held
            try:
                yield
            finally:
                held = set(getattr(_MIGRATION_LOCK_HELD, "paths", set()))
                held.discard(key)
                _MIGRATION_LOCK_HELD.paths = held
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


REQUIRED_SCHEMA: dict[str, set[str]] = {
    "people": {"id", "name", "organization_id", "site_id"},
    "events": {"id", "event_type", "timestamp", "event_uid", "correlation_id", "organization_id", "site_id", "device_id"},
    "attendance": {"id", "person_id", "date", "attendance_status", "organization_id", "site_id", "device_id"},
    "presence_sessions": {"id", "entity_id", "status", "last_seen_at", "organization_id", "site_id", "device_id"},
    "recognition_evidence": {"id", "entity_id", "decision", "observed_at", "organization_id", "site_id", "device_id"},
    "attendance_decisions": {"id", "decision_key", "decision", "observed_at", "organization_id", "site_id", "device_id"},
    "incidents": {"id", "status", "category", "severity", "organization_id", "site_id", "device_id"},
    "platform_audit_log": {"id", "action", "entity_type", "record_hash", "hash_version", "organization_id", "site_id", "device_id"},
    "platform_outbox": {"id", "event_id", "event_type", "payload_checksum", "status", "organization_id", "site_id", "device_id"},
    "edge_sync_devices": {"device_id", "last_sequence", "last_hash", "organization_id", "site_id"},
    "edge_sync_batches": {"batch_id", "device_id", "last_sequence", "last_hash", "body_checksum"},
    "edge_sync_events": {"event_id", "device_id", "sequence", "payload_json", "record_hash"},
    "schema_migrations": {"version", "description", "checksum", "applied_at"},
    "schema_migration_state": {"schema_name", "version", "checksum", "schema_fingerprint", "applied_at"},
    "schema_migration_history": {"id", "schema_name", "version", "checksum", "status", "applied_at"},
    "schema_migration_runs": {"id", "run_id", "schema_name", "target_version", "status", "occurred_at"},
}


def dict_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@contextmanager
def get_connection(path=None):
    # Resolve the configured path at call time so tests and local deployments
    # can swap databases without a stale default argument.
    path = path or DATABASE_PATH
    started = time.perf_counter()
    con = None
    try:
      con = connect_database(path, timeout=10)
      _metric_increment("connections_opened")
      con.execute("PRAGMA busy_timeout=10000")
      con.execute("PRAGMA foreign_keys=ON")
      con.execute(f"PRAGMA synchronous={SQLITE_SYNCHRONOUS}")
      # secure_delete prevents ordinary deleted pages from retaining sensitive
      # attendance or identity fragments. Strict modes enable it by default.
      con.execute(f"PRAGMA secure_delete={SQLITE_SECURE_DELETE}")
      con.execute("PRAGMA temp_store=MEMORY")
      con.execute("PRAGMA wal_autocheckpoint=1000")
      try:
          con.execute("PRAGMA journal_mode=WAL")
      except sqlite3.OperationalError:
          pass
      yield con
    except sqlite3.Error as exc:
      _metric_increment("operation_errors")
      if _is_lock_error(exc):
          _metric_increment("lock_errors")
      if con is not None:
          con.rollback()
      raise DatabaseError(str(exc)) from exc
    except DatabaseError:
      _metric_increment("connection_errors")
      raise
    except Exception:
      _metric_increment("operation_errors")
      if con is not None:
          con.rollback()
      raise
    finally:
      if con is not None:
          con.close()
          _metric_increment("connections_closed")
          _metric_latency(_CONNECTION_LATENCIES_MS, (time.perf_counter() - started) * 1000.0)


@contextmanager
def transaction(path=None, *, immediate: bool = False):
    """Run a unit of work with explicit commit/rollback semantics.

    Callers that perform multiple related writes should use this boundary
    rather than relying on connection close behavior. ``immediate=True`` is
    useful for idempotent command or attendance updates that must reserve the
    write lock before reading and deciding.
    """
    with get_connection(path) as con:
        started = time.perf_counter()
        try:
            con.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            _metric_increment("transactions_started")
        except sqlite3.Error as exc:
            _metric_increment("operation_errors")
            if _is_lock_error(exc):
                _metric_increment("lock_errors")
            raise DatabaseError(str(exc)) from exc
        try:
            yield con
            con.commit()
            _metric_increment("transactions_committed")
        except Exception as exc:
            con.rollback()
            _metric_increment("transactions_rolled_back")
            if _is_lock_error(exc):
                _metric_increment("lock_errors")
            raise
        finally:
            _metric_latency(_TRANSACTION_LATENCIES_MS, (time.perf_counter() - started) * 1000.0)


def fetch_all(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with get_connection() as con:
        return [dict(row) for row in con.execute(sql, tuple(params)).fetchall()]


def fetch_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with get_connection() as con:
        return dict_row(con.execute(sql, tuple(params)).fetchone())


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with get_connection() as con:
        cur = con.execute(sql, tuple(params))
        con.commit()
        return int(cur.lastrowid or cur.rowcount or 0)


def table_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with get_connection() as con:
        rows = con.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
        ).fetchall()
        for row in rows:
            name = row["name"]
            quoted_name = str(name).replace('"', '""')
            counts[name] = con.execute(f"select count(*) as c from \"{quoted_name}\"").fetchone()["c"]
    return counts


def schema_contract_issues(con: sqlite3.Connection) -> list[str]:
    """Check application-critical tables and columns without mutating data."""
    issues: list[str] = []
    for table, required_columns in REQUIRED_SCHEMA.items():
        exists = con.execute(
            "select 1 from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()
        if not exists:
            issues.append(f"missing table: {table}")
            continue
        actual = {row[1] for row in con.execute(f"pragma table_info(\"{table}\")").fetchall()}
        for column in sorted(required_columns - actual):
            issues.append(f"missing column: {table}.{column}")
    return issues


def database_health(path=None, *, include_counts: bool = False) -> dict[str, Any]:
    """Return a bounded operational health report for a SQLite database.

    This deliberately uses ``quick_check`` for the request path. Full
    ``integrity_check`` remains a backup/maintenance concern because it can
    scan every index and table on a large local database.
    """
    resolved = Path(path or DATABASE_PATH).expanduser().resolve()
    encryption = database_encryption_status()
    if not resolved.exists():
        return {
            "status": "missing",
            "connected": False,
            "path": str(resolved),
            "size_bytes": 0,
            "quick_check": "not_run",
            "foreign_key_violations": 0,
            "journal_mode": "not_run",
            "schema_version": None,
            "migration_version": None,
            "encryption": encryption,
            "metrics": database_metrics(),
        }
    try:
        with get_connection(resolved) as con:
            quick = str(con.execute("pragma quick_check").fetchone()[0])
            fk_rows = con.execute("pragma foreign_key_check").fetchall()
            journal_mode = str(con.execute("pragma journal_mode").fetchone()[0])
            schema_version = int(con.execute("pragma user_version").fetchone()[0])
            page_size = int(con.execute("pragma page_size").fetchone()[0] or 0)
            page_count = int(con.execute("pragma page_count").fetchone()[0] or 0)
            freelist_pages = int(con.execute("pragma freelist_count").fetchone()[0] or 0)
            wal_path = Path(f"{resolved}-wal")
            shm_path = Path(f"{resolved}-shm")
            database_size_bytes = resolved.stat().st_size
            wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0
            free_bytes = shutil.disk_usage(resolved.parent).free
            capacity_issues: list[str] = []
            if database_size_bytes >= DATABASE_MAX_BYTES:
                capacity_issues.append("database_size_limit_exceeded")
            elif database_size_bytes >= DATABASE_WARN_BYTES:
                capacity_issues.append("database_size_near_limit")
            if wal_size_bytes >= DATABASE_WAL_MAX_BYTES:
                capacity_issues.append("wal_size_limit_exceeded")
            if free_bytes < DATABASE_MIN_FREE_BYTES:
                capacity_issues.append("filesystem_free_space_low")
            capacity_status = "CRITICAL" if any(
                item.endswith("exceeded") or item == "filesystem_free_space_low"
                for item in capacity_issues
            ) else "WARN" if capacity_issues else "OK"
            migration_version = None
            migration_states: list[dict[str, Any]] = []
            migration_history: list[dict[str, Any]] = []
            migration_runs: list[dict[str, Any]] = []
            if con.execute(
                "select 1 from sqlite_master where type='table' and name='schema_migrations'"
            ).fetchone():
                migration_version = con.execute(
                    "select max(version) from schema_migrations"
                ).fetchone()[0]
            if con.execute(
                "select 1 from sqlite_master where type='table' and name='schema_migration_state'"
            ).fetchone():
                state_columns = {
                    row[1] for row in con.execute(
                        "pragma table_info(schema_migration_state)"
                    ).fetchall()
                }
                fingerprint_column = ", schema_fingerprint" if "schema_fingerprint" in state_columns else ""
                migration_states = [
                    dict(row) for row in con.execute(
                        f"select schema_name, version, description, checksum{fingerprint_column}, applied_at "
                        "from schema_migration_state order by schema_name"
                    ).fetchall()
                ]
            if con.execute(
                "select 1 from sqlite_master where type='table' and name='schema_migration_history'"
            ).fetchone():
                migration_history = [
                    dict(row) for row in con.execute(
                        "select schema_name, version, checksum, status, applied_at, process_id, host_name "
                        "from schema_migration_history order by id desc limit 50"
                    ).fetchall()
                ]
            if con.execute(
                "select 1 from sqlite_master where type='table' and name='schema_migration_runs'"
            ).fetchone():
                migration_runs = [
                    dict(row) for row in con.execute(
                        "select run_id, schema_name, target_version, status, error_code, occurred_at, process_id, host_name "
                        "from schema_migration_runs order by id desc limit 50"
                    ).fetchall()
                ]
            schema_issues = schema_contract_issues(con)
            consistency = consistency_issues(con)
            schema_object_contract_issues = schema_object_issues(con)
            migration_ledger_issues = migration_issues(con)
            migration_state_issues = schema_state_issues(con, ("platform", "outbox"))
            latest_run_status: dict[str, str] = {}
            # Rows are newest-first. Only the terminal/current status for each
            # run is actionable; the earlier committed `started` marker is
            # expected after a successful `applied` marker.
            for row in migration_runs:
                latest_run_status.setdefault(str(row.get("run_id")), str(row.get("status")))
            migration_run_issues = [
                f"migration run {run_id} is {status}"
                for run_id, status in latest_run_status.items()
                if status != "applied"
            ]
            audit = verify_audit_chain(con, "platform_audit_log")
            result: dict[str, Any] = {
                "status": "healthy" if quick.lower() == "ok" and not fk_rows and not schema_issues and not schema_object_contract_issues and not consistency and not migration_ledger_issues and not migration_state_issues and not migration_run_issues and audit.get("ok") and capacity_status == "OK" else "degraded",
                "connected": True,
                "path": str(resolved),
                "size_bytes": database_size_bytes,
                "page_size": page_size,
                "page_count": page_count,
                "freelist_pages": freelist_pages,
                "estimated_database_bytes": page_size * page_count,
                "wal_size_bytes": wal_size_bytes,
                "shm_size_bytes": shm_path.stat().st_size if shm_path.exists() else 0,
                "capacity": {
                    "status": capacity_status,
                    "database_size_bytes": database_size_bytes,
                    "database_warn_bytes": DATABASE_WARN_BYTES,
                    "database_max_bytes": DATABASE_MAX_BYTES,
                    "wal_size_bytes": wal_size_bytes,
                    "wal_max_bytes": DATABASE_WAL_MAX_BYTES,
                    "free_bytes": free_bytes,
                    "min_free_bytes": DATABASE_MIN_FREE_BYTES,
                    "issues": capacity_issues,
                },
                "quick_check": quick,
                "foreign_key_violations": len(fk_rows),
                "journal_mode": journal_mode,
                "schema_version": schema_version,
                "migration_version": migration_version,
                "migration_states": migration_states,
                "migration_history": migration_history,
                "migration_runs": migration_runs,
                "schema_issues": schema_issues,
                "consistency_issues": consistency,
                "schema_object_issues": schema_object_contract_issues,
                "migration_ledger_issues": migration_ledger_issues,
                "migration_state_issues": migration_state_issues,
                "migration_run_issues": migration_run_issues,
                "audit": audit,
                "encryption": encryption,
                "metrics": database_metrics(),
            }
            if include_counts:
                result["tables"] = {
                    row["name"]: con.execute(
                        f"select count(*) as c from \"{row['name'].replace(chr(34), chr(34) * 2)}\""
                    ).fetchone()["c"]
                    for row in con.execute(
                        "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
                    ).fetchall()
                }
            # Leave the connection context before sampling metrics so the
            # report includes the health query's close and latency counters.
        result["metrics"] = database_metrics()
        return result
    except (OSError, sqlite3.Error, DatabaseError) as exc:
        return {
            "status": "error",
            "connected": False,
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size if resolved.exists() else 0,
            "encryption": encryption,
            "metrics": database_metrics(),
            "error": str(exc),
        }


def database_maintenance(
    path=None,
    *,
    full_integrity: bool = False,
    checkpoint: bool = False,
    optimize: bool = True,
    repair_schema: bool = False,
) -> dict[str, Any]:
    """Run explicit, bounded SQLite maintenance outside request health checks."""
    resolved = Path(path or DATABASE_PATH).expanduser().resolve()
    if not resolved.exists():
        raise DatabaseError("Database does not exist.")
    result: dict[str, Any] = {
        "path": str(resolved),
        "full_integrity_requested": bool(full_integrity),
        "checkpoint_requested": bool(checkpoint),
        "optimize_requested": bool(optimize),
        "repair_schema_requested": bool(repair_schema),
    }
    if repair_schema:
        # Import lazily to avoid a module cycle: platform_schema depends on
        # this database connection layer.
        from .platform_schema import ensure_platform_schema
        ensure_platform_schema(resolved)
        result["schema_repaired"] = True
    with get_connection(resolved) as con:
        if full_integrity:
            result["integrity_check"] = str(con.execute("pragma integrity_check").fetchone()[0])
        if checkpoint:
            row = con.execute("pragma wal_checkpoint(passive)").fetchone()
            result["wal_checkpoint"] = {
                "busy": int(row[0] or 0),
                "log_pages": int(row[1] or 0),
                "checkpointed_pages": int(row[2] or 0),
            }
        if optimize:
            con.execute("pragma optimize")
            result["optimized"] = True
    result["health"] = database_health(resolved)
    return result
