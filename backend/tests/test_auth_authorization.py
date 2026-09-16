from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import config, database
from backend.main import app
from backend.platform_schema import ensure_platform_schema
from backend.security import has_permission
from backend.services import auth_service


@pytest.fixture()
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    ensure_platform_schema()
    yield


def add_user(username: str, role: str, password: str = "correct horse") -> None:
    database.execute(
        "insert into platform_users (username, password_hash, role) values (?, ?, ?)",
        [username, auth_service.hash_password(password), role],
    )


def login_client(username: str = "operator") -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": username, "password": "correct horse"})
    assert response.status_code == 200
    assert "access_token" not in response.json()
    assert client.cookies.get(config.SESSION_COOKIE_NAME)
    assert client.cookies.get(config.CSRF_COOKIE_NAME)
    return client


def test_cookie_login_rotates_sessions_and_returns_no_token(auth_db):
    add_user("operator", "operator")
    first = auth_service.login("operator", "correct horse")
    second = auth_service.login("operator", "correct horse")
    assert auth_service.resolve_session(first["access_token"]) is None
    assert auth_service.resolve_session(second["access_token"])["username"] == "operator"

    client = login_client()
    assert client.get("/api/auth/me").status_code == 200


def test_cookie_authenticated_writes_require_csrf(auth_db):
    add_user("reviewer", "security-reviewer")
    client = login_client("reviewer")
    payload = {"action": "dismiss", "note": "test"}
    assert client.post("/api/events/999/review", json=payload).status_code == 403
    response = client.post(
        "/api/events/999/review",
        json=payload,
        headers={"X-CSRF-Token": client.cookies.get(config.CSRF_COOKIE_NAME)},
    )
    assert response.status_code == 404


def test_logout_all_revokes_current_session_and_audits(auth_db):
    add_user("operator", "operator")
    client = login_client()
    csrf = client.cookies.get(config.CSRF_COOKIE_NAME)
    response = client.post("/api/auth/logout-all", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200
    assert response.json()["revoked_sessions"] == 1
    assert database.fetch_one("select 1 from platform_audit_log where action='auth.logout_all'")


def test_role_permissions_separate_operational_and_administrative_actions():
    assert has_permission("viewer", "live.view")
    assert not has_permission("viewer", "attendance.manual")
    assert has_permission("operator", "attendance.manual")
    assert not has_permission("operator", "biometric.delete")
    assert has_permission("attendance-admin", "attendance.manage")
    assert not has_permission("attendance-admin", "biometric.delete")
    assert has_permission("biometric-admin", "biometric.delete")
    assert has_permission("security-reviewer", "security.review")
    assert not has_permission("security-reviewer", "attendance.manage")
    assert has_permission("system-admin", "system.manage")


def test_operator_cannot_change_people_profile(auth_db):
    add_user("operator", "operator")
    person_id = database.execute("insert into people (name, role) values (?, ?)", ["Ada", "Student"])
    client = login_client()
    response = client.patch(
        f"/api/people/{person_id}",
        json={"notes": "attempt"},
        headers={"X-CSRF-Token": client.cookies.get(config.CSRF_COOKIE_NAME)},
    )
    assert response.status_code == 403


def test_sensitive_routes_have_permission_dependencies():
    expected = {
        ("GET", "/api/live/frame"),
        ("GET", "/api/people"),
        ("GET", "/api/people/{person_id}"),
        ("GET", "/api/attendance"),
        ("GET", "/api/attendance/export"),
        ("POST", "/api/commands"),
        ("POST", "/api/enrollment/start"),
        ("DELETE", "/api/people/{person_name}"),
        ("POST", "/api/events/{event_id}/review"),
        ("POST", "/api/incidents/{incident_id}/review"),
        ("GET", "/api/system/performance"),
    }
    found = {(method, route.path): route for route in app.routes for method in getattr(route, "methods", set())}
    for key in expected:
        route = found[key]
        assert route.dependant.dependencies, f"{key} has no authorization dependency"


def test_websocket_rejects_missing_authentication_outside_exhibition(auth_db, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME_MODE", "pilot")
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live"):
            pass


def test_websocket_rejects_untrusted_origin(auth_db):
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live", headers={"origin": "https://untrusted.example"}):
            pass
