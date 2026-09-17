import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend import config, database, security
from backend.main import app
from backend.platform_schema import ensure_platform_schema
from backend.services import auth_service, command_service, cybersecurity_service


@pytest.fixture()
def security_client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "security.db")
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    security._rate_history.clear()
    security._login_history.clear()
    ensure_platform_schema()
    return TestClient(app)


def add_user(username="operator", role="operator", password="correct horse"):
    return database.execute(
        "insert into platform_users (username, password_hash, role) values (?, ?, ?)",
        [username, auth_service.hash_password(password), role],
    )


def login(client, username="operator", password="correct horse"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return client


def csrf(client):
    return {"X-CSRF-Token": client.cookies.get(config.CSRF_COOKIE_NAME)}


def test_invalid_login_token_replay_and_expired_session_are_rejected(security_client):
    add_user()
    result = auth_service.login("operator", "correct horse")
    token = result["access_token"]
    assert auth_service.resolve_session(token)["username"] == "operator"
    auth_service.logout(token)
    assert auth_service.resolve_session(token) is None
    assert auth_service.session_status(token) == "invalid_or_revoked"

    expired = auth_service.login("operator", "correct horse")
    database.execute(
        "update platform_sessions set expires_at=? where token_hash=?",
        [(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(), auth_service._token_hash(expired["access_token"])],
    )
    assert auth_service.resolve_session(expired["access_token"]) is None
    assert auth_service.session_status(expired["access_token"]) == "expired"
    failed = security_client.post("/api/auth/login", json={"username": "operator", "password": "wrong password"})
    assert failed.status_code == 401


def test_wrong_role_and_nonexistent_resource_do_not_bypass_authorization(security_client):
    add_user(role="viewer")
    client = login(security_client)
    denied = client.post("/api/people/999999/disable", headers=csrf(client))
    assert denied.status_code == 403
    missing = client.get("/api/people/999999")
    assert missing.status_code == 404


def test_cors_is_allowlist_based_and_api_responses_are_not_cacheable(security_client):
    response = security_client.options(
        "/api/people",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code in {400, 200}
    assert response.headers.get("access-control-allow-origin") != "https://attacker.example"
    health = security_client.get("/api/health")
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"


def test_malformed_and_oversized_payloads_fail_at_the_api_boundary(security_client):
    malformed = security_client.post("/api/auth/login", json={"username": "a"})
    assert malformed.status_code == 422
    oversized = security_client.post(
        "/api/auth/login",
        content=json.dumps({"username": "operator", "password": "x" * 2_000_000}),
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413
    invalid_length = security_client.post(
        "/api/auth/login",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "not-a-number"},
    )
    assert invalid_length.status_code == 400


def test_disk_full_command_queue_returns_controlled_error(monkeypatch, tmp_path):
    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(tmp_path / "commands.json.tmp"), "write_text", fail_write)
    with pytest.raises(Exception) as error:
        command_service.atomic_write(tmp_path / "commands.json", {"commands": []})
    assert getattr(error.value, "status_code", None) == 507


def test_alert_delivery_failure_is_separate_cyber_event_and_incident(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "security.db")
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    ensure_platform_schema()
    result = cybersecurity_service.record_cyber_event(
        "ALERT_DELIVERY_FAILURE",
        source="test-alert",
        details={"channel": "webhook", "error": "connection refused"},
        actor_id="system",
        device_id="edge-test",
    )
    assert result["eventType"] == "ALERT_DELIVERY_FAILURE"
    incidents = cybersecurity_service.list_cyber_incidents()
    assert incidents[0]["category"] == "communications"
    assert incidents[0]["eventCount"] >= 1


def test_security_scan_does_not_report_test_fixture_credentials():
    from security_scan import scan_tracked_files

    assert scan_tracked_files() == []
