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
