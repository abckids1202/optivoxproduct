from __future__ import annotations

import json

import pytest

from backend import config, database
from backend.platform_schema import ensure_platform_schema
from backend.services import auth_service, cybersecurity_service as cyber


@pytest.fixture()
def cyber_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "cyber.db")
    monkeypatch.setattr(config, "RUNTIME_MODE", "development")
    ensure_platform_schema()
    yield tmp_path


def test_events_group_by_context_and_stay_out_of_physical_incidents(cyber_db):
    first = cyber.record_cyber_event(
        "FAILED_LOGIN_BURST", source="authentication", actor_id="alice",
        actor_type="system", ip_address="10.0.0.5", user_agent="test-agent",
        occurred_at="2026-09-16T10:00:00+00:00", details={"username": "alice"},
    )
    second = cyber.record_cyber_event(
        "FAILED_LOGIN_BURST", source="authentication", actor_id="alice",
        actor_type="system", ip_address="10.0.0.5", user_agent="test-agent",
        occurred_at="2026-09-16T10:02:00+00:00", details={"username": "alice"},
    )
    separate = cyber.record_cyber_event(
        "FAILED_LOGIN_BURST", source="authentication", actor_id="alice",
        actor_type="system", ip_address="10.0.0.6", user_agent="test-agent",
        occurred_at="2026-09-16T10:02:00+00:00", details={"username": "alice"},
    )

    assert first["correlationId"] != second["correlationId"]
    assert first["incidentId"] == second["incidentId"]
    assert separate["incidentId"] != first["incidentId"]
    assert database.fetch_one("select count(*) as count from incidents")["count"] == 0
    grouped = cyber.get_cyber_incident(first["incidentId"])
    assert grouped["eventCount"] == 2
    assert grouped["alertCount"] == 1
    assert grouped["ipAddress"] == "10.0.0.5"
    assert grouped["userAgent"] == "test-agent"
    assert database.fetch_one("select count(*) as count from platform_outbox")["count"] == 3


def test_dedupe_key_makes_runtime_events_idempotent(cyber_db):
    kwargs = dict(
        source="runtime", device_id="cam_0", occurred_at="2026-09-16T10:00:00+00:00",
        details={"state": "DISCONNECTED"}, dedupe_key="runtime:camera:one",
    )
    first = cyber.record_cyber_event("CAMERA_DISCONNECT", **kwargs)
    second = cyber.record_cyber_event("CAMERA_DISCONNECT", **kwargs)
    assert first["id"] == second["id"]
    assert database.fetch_one("select count(*) as count from cybersecurity_events")["count"] == 1


def test_cyber_event_and_audit_record_commit_atomically(cyber_db, monkeypatch):
    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit storage unavailable")

    monkeypatch.setattr("backend.services.audit_service.append_audit_record", fail_audit)
    with pytest.raises(RuntimeError, match="audit storage unavailable"):
        cyber.record_cyber_event("RUNTIME_CRASH", source="runtime", details={"test": True})
    assert database.fetch_one("select count(*) as count from cybersecurity_events")["count"] == 0
    assert database.fetch_one("select count(*) as count from cybersecurity_incidents")["count"] == 0
    assert database.fetch_one("select count(*) as count from platform_outbox")["count"] == 0


def test_review_escalation_and_resolution_are_traceable(cyber_db):
    event = cyber.record_cyber_event("UNAUTHORIZED_COMMAND", source="authorization", actor_id="operator-1", details={"command": "delete_person"})
    acknowledged = cyber.review_cyber_incident(event["incidentId"], "acknowledge", "triaged", actor_id="reviewer")
    escalated = cyber.review_cyber_incident(event["incidentId"], "escalate", "needs administrator", actor_id="reviewer")
    resolved = cyber.review_cyber_incident(event["incidentId"], "resolve", "credentials rotated", actor_id="admin")
    assert acknowledged["status"] == "acknowledged"
    assert escalated["status"] == "escalated"
    assert resolved["status"] == "resolved"
    assert [row["action"] for row in resolved["reviews"]] == ["resolve", "escalate", "acknowledge"]
    assert resolved["resolvedBy"] == "admin"


def test_alert_delivery_failure_is_correlated_and_exposed(cyber_db):
    database.execute(
        "insert into alert_log(channel, event_type, target, status, error, timestamp) values (?, ?, ?, ?, ?, ?)",
        ["webhook", "ZONE_INTRUSION", "https://blocked.example", "failed", "unexpected outbound destination", "2026-09-16T10:00:00+00:00"],
    )
    failures = cyber.alert_failures()
    assert failures
    assert failures[0]["eventType"] == "UNEXPECTED_OUTBOUND_DESTINATION"
    assert failures[0]["details"]["channel"] == "webhook"
    assert failures[0]["evidenceRef"].startswith("alert_log:")


def test_runtime_health_failure_is_ingested_once(cyber_db, monkeypatch, tmp_path):
    health_path = tmp_path / "health_events.jsonl"
    health_path.write_text(json.dumps({
        "type": "CAMERA_HEALTH", "state": "DISCONNECTED",
        "timestamp": "2026-09-16T10:00:00+00:00", "camera_id": "cam_0",
        "recorded_at": "2026-09-16T10:00:01+00:00",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(cyber, "HEALTH_EVENTS_PATH", health_path)
    first = cyber.sync_operational_security_sources()
    second = cyber.sync_operational_security_sources()
    assert first["runtime"] == 1
    assert second["runtime"] == 0
    event = database.fetch_one("select * from cybersecurity_events where event_type='CAMERA_DISCONNECT'")
    assert event["source"] == "runtime"


def test_active_sessions_never_expose_session_token_hash(cyber_db):
    database.execute(
        "insert into platform_users(username, password_hash, role) values (?, ?, ?)",
        ["operator", auth_service.hash_password("correct horse"), "operator"],
    )
    auth_service.login("operator", "correct horse", client_ip="127.0.0.1", user_agent="test")
    sessions = cyber.active_sessions()
    assert len(sessions) == 1
    assert "token_hash" not in sessions[0]
    assert sessions[0]["client_ip"] == "127.0.0.1"
