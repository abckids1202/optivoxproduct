from __future__ import annotations

from starlette.requests import Request

from backend import config
from backend import security
from backend.services import runtime_service
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


def test_performance_report_returns_not_measured_without_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_service, "PERFORMANCE_SUMMARY_PATH", tmp_path / "missing.json")
    result = runtime_service.performance_report()
    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["end_to_end_latency_ms"] is None


def test_tracked_secret_scan_is_clean():
    assert scan_tracked_files() == []


def test_readme_declares_the_canonical_runtime():
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "main.py" in readme
    assert "backend/" in readme
    assert "frontend/" in readme
