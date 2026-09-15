from fastapi.testclient import TestClient

from backend.main import app


def test_mutating_routes_require_configured_key(monkeypatch):
    monkeypatch.setenv("OPTIVOX_API_KEY", "test-secret")
    client = TestClient(app)
    response = client.post("/api/commands", json={"command": "save_snapshot", "payload": {}})
    assert response.status_code == 401


def test_configured_key_allows_operator_command(monkeypatch):
    monkeypatch.setenv("OPTIVOX_API_KEY", "test-secret")
    client = TestClient(app)
    response = client.post(
        "/api/commands",
        headers={"X-Optivox-Key": "test-secret", "Idempotency-Key": "test-security-command"},
        json={"command": "save_snapshot", "payload": {}},
    )
    assert response.status_code == 200
    assert response.json()["type"] == "save_snapshot"


def test_sensitive_read_routes_require_configured_key(monkeypatch):
    monkeypatch.setenv("OPTIVOX_API_KEY", "test-secret")
    client = TestClient(app)
    response = client.get("/api/people")
    assert response.status_code == 401


def test_login_attempts_are_rate_limited(monkeypatch):
    from backend import security

    security._login_history.clear()
    client = TestClient(app)
    responses = [client.post("/api/auth/login", json={"username": "missing", "password": "bad-password"}) for _ in range(11)]
    assert responses[-1].status_code == 429
    security._login_history.clear()
