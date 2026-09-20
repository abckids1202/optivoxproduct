from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import storage_service


def test_parallel_transactions_preserve_all_event_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "concurrent.db")
    ensure_platform_schema()

    def writer(worker: int) -> int:
        written = 0
        for sequence in range(20):
            with database.transaction(immediate=True) as con:
                con.execute(
                    """insert into events
                       (event_type, timestamp, event_uid, correlation_id)
                       values (?, ?, ?, ?)""",
                    (
                        "CONCURRENCY_TEST",
                        f"2026-09-18T10:00:{worker:02d}{sequence:02d}",
                        f"concurrency:{worker}:{sequence}",
                        f"correlation:{worker}:{sequence}",
                    ),
                )
            written += 1
        return written

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(writer, range(8)))

    assert sum(results) == 160
    assert database.fetch_one(
        "select count(*) as count from events where event_type='CONCURRENCY_TEST'"
    )["count"] == 160
    health = database.database_health()
    assert health["quick_check"] == "ok"
    assert health["foreign_key_violations"] == 0
    assert health["consistency_issues"] == []


def test_database_metrics_report_transactions_and_bounded_latency(tmp_path, monkeypatch):
    path = tmp_path / "metrics.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    ensure_platform_schema(path)
    database.reset_database_metrics()
    with database.transaction(path, immediate=True) as con:
        con.execute("create table metric_sample(value text)")
        con.execute("insert into metric_sample values ('ok')")
    try:
        with database.transaction(path, immediate=True) as con:
            con.execute("insert into metric_sample values ('rollback')")
            raise RuntimeError("expected rollback")
    except RuntimeError:
        pass
    database.fetch_one("select count(*) as count from metric_sample")
    metrics = database.database_metrics()
    assert metrics["connections_opened"] >= 3
    assert metrics["connections_closed"] == metrics["connections_opened"]
    assert metrics["transactions_started"] == 2
    assert metrics["transactions_committed"] == 1
    assert metrics["transactions_rolled_back"] == 1
    assert metrics["latency_ms"]["connection_p95"] is not None
    assert metrics["latency_ms"]["transaction_p95"] is not None


def test_parallel_schema_initialization_is_serialized(tmp_path, monkeypatch):
    path = tmp_path / "schema-race.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)

    def initialize(_: int) -> None:
        ensure_platform_schema(path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(initialize, range(6)))

    health = database.database_health(path)
    assert health["status"] == "healthy"
    assert health["migration_state_issues"] == []
    assert (tmp_path / "schema-race.db.schema.lock").exists()


def test_database_health_reports_capacity_pressure(tmp_path, monkeypatch):
    path = tmp_path / "capacity.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    monkeypatch.setattr(database, "DATABASE_WARN_BYTES", 1)
    monkeypatch.setattr(database, "DATABASE_MAX_BYTES", 2)
    monkeypatch.setattr(database, "DATABASE_WAL_MAX_BYTES", 2**40)
    monkeypatch.setattr(database, "DATABASE_MIN_FREE_BYTES", 1)
    ensure_platform_schema(path)
    health = database.database_health(path)
    assert health["capacity"]["status"] == "CRITICAL"
    assert "database_size_limit_exceeded" in health["capacity"]["issues"]
    assert health["status"] == "degraded"


def test_backup_recovery_status_requires_recent_valid_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", tmp_path / "backups")
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "development")

    missing = storage_service.backup_recovery_status(max_age_minutes=5)
    assert missing["status"] == "WARN"
    assert "no_valid_backup" in missing["issues"]

    database_path = tmp_path / "source.db"
    ensure_platform_schema(database_path)
    backup = storage_service.backup_database(source=database_path)
    fresh = storage_service.backup_recovery_status(max_age_minutes=5)
    assert fresh["status"] == "OK"
    assert fresh["latest_backup_name"] == Path(backup["path"]).name
    assert fresh["latest_backup_age_seconds"] is not None

    stale = storage_service.backup_recovery_status(
        max_age_minutes=5,
        now=datetime.now(timezone.utc) + timedelta(minutes=6),
    )
    assert stale["status"] == "WARN"
    assert "latest_backup_stale" in stale["issues"]


def test_backup_recovery_status_rejects_unreadable_latest_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", tmp_path / "backups")
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "development")
    database_path = tmp_path / "source.db"
    ensure_platform_schema(database_path)
    backup = storage_service.backup_database(source=database_path)
    backup_path = Path(backup["path"])
    payload = bytearray(backup_path.read_bytes())
    payload[0] ^= 0xFF
    backup_path.write_bytes(payload)
    manifest_path = Path(backup["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(bytes(payload)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status = storage_service.backup_recovery_status(max_age_minutes=5)
    assert status["status"] == "WARN"
    assert status["valid_count"] == 1
    assert "latest_backup_unreadable" in status["issues"]


def test_backup_recovery_drill_restores_into_isolated_temporary_database(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", tmp_path / "backups")
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "development")
    database_path = tmp_path / "source.db"
    ensure_platform_schema(database_path)
    backup = storage_service.backup_database(source=database_path)
    result = storage_service.rehearse_latest_backup()
    assert result["status"] == "success"
    assert result["backup_name"] == Path(backup["path"]).name
    assert result["integrity_check"] == "ok"
    assert result["foreign_key_violations"] == 0
    assert result["table_count"] > 0
    assert not any(path.name.startswith(".restore-drill-") for path in (tmp_path / "backups").iterdir())


def test_backup_recovery_drill_fails_without_verified_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", tmp_path / "backups")
    result = storage_service.rehearse_latest_backup()
    assert result == {"status": "failed", "reason": "no_valid_backup"}
