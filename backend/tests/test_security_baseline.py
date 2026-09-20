from __future__ import annotations

from starlette.requests import Request
import pytest

from backend import config
from backend import database
from backend import security
from backend.services import runtime_service
from backend.services import system_service
from security_scan import scan_tracked_files


def request_for(host: str = "127.0.0.1") -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/api/test",
        "headers": [],
        "client": (host, 12345),
        "server": (host, 8000),
        "scheme": "http",
    })


def clear_auth(monkeypatch):
    for name in (
        "OPTIVOX_API_KEY", "OPTIVOX_OPERATOR_KEY", "OPTIVOX_ADMIN_KEY",
        "OPTIVOX_ADMIN_USERNAME", "OPTIVOX_ADMIN_PASSWORD",
        "OPTIVOX_OPERATOR_USERNAME", "OPTIVOX_OPERATOR_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_development_loopback_bypass_remains_explicitly_available(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    assert security._guard(request_for(), None, None) == "local-operator"


def test_pilot_without_auth_fails_closed_even_on_loopback(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "pilot")
    try:
        security._guard(request_for(), None, None)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 503
        assert exc.detail["code"] == "AUTH_NOT_CONFIGURED"
    else:
        raise AssertionError("pilot mode must not trust loopback without authentication")


def test_production_configuration_requires_authentication(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "production")
    monkeypatch.setattr(config, "DEMO_MODE", False)
    issues = config.configuration_issues()
    assert any("require an API key" in issue for issue in issues)
    try:
        config.validate_runtime_configuration()
    except RuntimeError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("unsafe production configuration must fail closed")


def test_strict_modes_require_durable_sqlite_settings(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "production")
    monkeypatch.setattr(config, "SQLITE_SYNCHRONOUS", "NORMAL")
    monkeypatch.setattr(config, "SQLITE_SECURE_DELETE", "OFF")
    issues = config.configuration_issues()
    assert any("require OPTIVOX_SQLITE_SYNCHRONOUS=FULL" in issue for issue in issues)
    assert any("require OPTIVOX_SQLITE_SECURE_DELETE=ON" in issue for issue in issues)


def test_strict_modes_require_encrypted_backups(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "production")
    monkeypatch.setattr(config, "BACKUP_ENCRYPTION_ENABLED", False)
    monkeypatch.setattr(config, "BACKUP_ENCRYPTION_KEY", "")
    issues = config.configuration_issues()
    assert any("BACKUP_ENCRYPTION_ENABLED=true" in issue for issue in issues)
    assert any("BACKUP_ENCRYPTION_KEY" in issue for issue in issues)


def test_strict_modes_require_a_backup_owner(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "production")
    monkeypatch.setattr(config, "BACKUP_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(config, "EXTERNAL_BACKUP_CONFIRMED", False)
    issues = config.configuration_issues()
    assert any("backup scheduler or" in issue for issue in issues)
    monkeypatch.setattr(config, "EXTERNAL_BACKUP_CONFIRMED", True)
    issues = config.configuration_issues()
    assert not any("backup scheduler or" in issue for issue in issues)


def test_strict_modes_require_encrypted_evidence(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "pilot")
    monkeypatch.setattr(config, "EVIDENCE_ENCRYPTION_ENABLED", False)
    monkeypatch.setattr(config, "EVIDENCE_ENCRYPTION_KEY", "")
    issues = config.configuration_issues()
    assert any("EVIDENCE_ENCRYPTION_ENABLED=true" in issue for issue in issues)
    assert any("EVIDENCE_ENCRYPTION_KEY" in issue for issue in issues)


def test_strict_modes_require_encrypted_live_database(monkeypatch):
    clear_auth(monkeypatch)
    monkeypatch.setattr(config, "RUNTIME_MODE", "pilot")
    monkeypatch.setattr(config, "DATABASE_ENCRYPTION_ENABLED", False)
    monkeypatch.setattr(config, "DATABASE_ENCRYPTION_KEY", "")
    issues = config.configuration_issues()
    assert any("DATABASE_ENCRYPTION_ENABLED=true" in issue for issue in issues)
    monkeypatch.setattr(config, "DATABASE_ENCRYPTION_ENABLED", True)
    issues = config.configuration_issues()
    assert any("DATABASE_ENCRYPTION_KEY" in issue for issue in issues)


def test_database_encryption_fails_closed_without_sqlcipher_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(database, "DATABASE_ENCRYPTION_KEY", "database-key-" + "x" * 32)
    monkeypatch.setattr(database, "_load_sqlcipher_driver", lambda: None)
    with pytest.raises(database.DatabaseError, match="SQLCipher"):
        database.connect_database(tmp_path / "encrypted.db")


def test_encrypted_backup_key_has_a_minimum_length(monkeypatch):
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    monkeypatch.setattr(config, "BACKUP_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(config, "BACKUP_ENCRYPTION_KEY", "too-short")
    assert any("at least 32 characters" in issue for issue in config.configuration_issues())


def test_performance_report_returns_not_measured_without_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_service, "PERFORMANCE_SUMMARY_PATH", tmp_path / "missing.json")
    result = runtime_service.performance_report()
    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["end_to_end_latency_ms"] is None


def test_performance_report_preserves_measured_end_to_end_latency(monkeypatch, tmp_path):
    summary_path = tmp_path / "performance.json"
    summary_path.write_text(
        '{"latency_ms":{"end_to_end_p95":245.5},'
        '"benchmark":{"ended_at":"2099-01-01T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_service, "PERFORMANCE_SUMMARY_PATH", summary_path)
    result = runtime_service.performance_report()
    assert result["measurement_status"] == "FRESH"
    assert result["end_to_end_latency_ms"] == 245.5


def test_performance_report_exposes_consumed_and_stale_frame_age(monkeypatch, tmp_path):
    summary_path = tmp_path / "performance-age.json"
    summary_path.write_text(
        '{"latency_ms":{"frame_age_p95":80.0,"frame_age_consumed_p95":125.0,'
        '"stale_frame_age_p95":410.0},'
        '"benchmark":{"ended_at":"2099-01-01T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_service, "PERFORMANCE_SUMMARY_PATH", summary_path)
    result = runtime_service.performance_report()
    assert result["frame_age_consumed_p95_ms"] == 125.0
    assert result["stale_frame_age_p95_ms"] == 410.0


def test_performance_report_preserves_partial_benchmark_status(monkeypatch, tmp_path):
    summary_path = tmp_path / "partial-performance.json"
    summary_path.write_text(
        '{"measurement_status":"PARTIAL","collection_quality":"PARTIAL",'
        '"benchmark":{"ended_at":"2099-01-01T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_service, "PERFORMANCE_SUMMARY_PATH", summary_path)

    result = runtime_service.performance_report()

    assert result["measurement_status"] == "PARTIAL"
    assert result["collection_quality"] == "PARTIAL"


def test_tracked_secret_scan_is_clean():
    assert scan_tracked_files() == []


def test_readme_declares_the_canonical_runtime():
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "main.py" in readme
    assert "backend/" in readme
    assert "frontend/" in readme


def test_security_zone_status_reports_override_without_biometric_data(monkeypatch, tmp_path):
    zone_path = tmp_path / "security_zones.json"
    zone_path.write_text(
        '{"schema_version": 1, "zones": [{"id": "lab", "severity": 3}], "updated_at": "2026-09-17T10:00:00+00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(system_service, "SECURITY_ZONES_PATH", zone_path)

    result = system_service.security_zones_status()

    assert result["source"] == "runtime_override"
    assert result["zones"] == [{"id": "lab", "severity": 3}]
    assert "embedding" not in str(result).lower()


def test_security_zone_status_rejects_wrong_override_shape(monkeypatch, tmp_path):
    zone_path = tmp_path / "security_zones.json"
    zone_path.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(system_service, "SECURITY_ZONES_PATH", zone_path)

    result = system_service.security_zones_status()

    assert result == {"source": "invalid_override", "zones": [], "updated_at": None}
