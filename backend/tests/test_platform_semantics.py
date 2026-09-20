import json
from datetime import datetime, timedelta

import pytest

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import attendance_service, command_service, event_service, incident_service, outbox_service, people_service


@pytest.fixture()
def platform_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    yield


def add_person(name, metadata=None):
    return database.execute(
        "insert into people (name, role, metadata_json, created_at, updated_at) values (?, ?, ?, datetime('now'), datetime('now'))",
        [name, "Student", json.dumps(metadata or {})],
    )


def test_metadata_and_roster_semantics(platform_db):
    first = add_person("Ada", {"class": "10A", "subjects": ["Math", None], "active": True})
    add_person("Grace", {"class_name": "10B", "subjects": "Physics"})
    today = datetime.now().date().isoformat()
    database.execute("insert into attendance (person_id, date, clock_in, notes) values (?, ?, datetime('now'), 'automatic')", [first, today])

    people = people_service.list_people()
    assert people[0]["className"] == "10A"
    assert people[0]["subjects"] == ["Math"]
    summary = attendance_service.attendance_summary()
    assert summary["total"] == 2
    assert summary["present"] == 1
    assert summary["not_yet_detected"] == 1


def test_operational_tables_have_deployment_scope_defaults(platform_db):
    scoped_tables = (
        "people", "events", "attendance", "absence_records",
        "attendance_schedules", "presence_sessions", "recognition_evidence",
        "enrollment_operations", "attendance_decisions", "liveness_challenges",
        "incidents", "alert_log", "platform_audit_log",
        "platform_outbox",
        "cybersecurity_events", "cybersecurity_incidents",
        "cybersecurity_incident_alerts", "cybersecurity_reviews",
        "edge_sync_devices", "edge_sync_batches", "edge_sync_events",
    )
    for table in scoped_tables:
        columns = {
            row["name"] for row in database.fetch_all(f"pragma table_info({table})")
        }
        assert {"organization_id", "site_id", "device_id"} <= columns, table

    person_id = add_person("Scoped person")
    database.execute(
        "insert into events (event_type, timestamp) values (?, datetime('now'))",
        ["HEALTH_CHECK"],
    )
    row = database.fetch_one(
        "select organization_id, site_id, device_id from people where id=?",
        [person_id],
    )
    event = database.fetch_one(
        "select organization_id, site_id, device_id from events order by id desc limit 1"
    )
    assert row["organization_id"] == "local-organization"
    assert row["site_id"] == "local-site"
    assert row["device_id"] == "local-edge-cam-0"
    assert dict(event) == {
        "organization_id": "local-organization",
        "site_id": "local-site",
        "device_id": "local-edge-cam-0",
    }


def test_platform_reads_do_not_cross_site_boundaries(platform_db):
    current_id = add_person("Current site")
    other_id = database.execute(
        "insert into people (name, role, metadata_json, organization_id, site_id, device_id, created_at, updated_at) values (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        ["Other site", "Student", "{}", "other-org", "other-site", "other-device"],
    )
    today = datetime.now().date().isoformat()
    database.execute(
        "insert into attendance (person_id, date, clock_in, organization_id, site_id, device_id) values (?, ?, datetime('now'), ?, ?, ?)",
        [other_id, today, "other-org", "other-site", "other-device"],
    )
    people = people_service.list_people()
    assert [person["id"] for person in people] == [current_id]
    with pytest.raises(Exception) as error:
        people_service.get_person(other_id)
    assert getattr(error.value, "status_code", None) == 404
    assert attendance_service.list_attendance() == []


def test_disabling_person_preserves_metadata(platform_db):
    person_id = add_person("Ada", {"class": "10A", "subjects": ["Math"], "active": True})
    person = people_service.set_enabled(person_id, False)
    assert person["active"] is False
    raw = database.fetch_one("select metadata_json from people where id=?", [person_id])
    assert json.loads(raw["metadata_json"]) == {"class": "10A", "subjects": ["Math"], "active": False}


def test_attendance_correction_is_recorded(platform_db):
    person_id = add_person("Ada")
    corrected = attendance_service.correct_attendance(
        person_id,
        "2026-08-21",
        "2026-08-21T08:10:00",
        "2026-08-21T15:30:00",
        10,
        "Teacher confirmed late arrival",
    )
    assert corrected["clock_in"] == "2026-08-21T08:10:00"
    assert corrected["work_minutes"] == 440
    audit = database.fetch_one("select action from platform_audit_log where action='attendance.correct'")
    correction = database.fetch_one("select reason from attendance_corrections where person_id=?", [person_id])
    assert audit["action"] == "attendance.correct"
    assert correction["reason"] == "Teacher confirmed late arrival"


def test_event_review_and_incident_grouping(platform_db):
    now = datetime.now().replace(microsecond=0)
    first = database.execute(
        "insert into events (event_type, severity, details_json, timestamp) values (?, ?, ?, ?)",
        ["DANGER_OBJECT", 2, '{"message":"object"}', now.isoformat()],
    )
    database.execute(
        "insert into events (event_type, severity, details_json, timestamp) values (?, ?, ?, ?)",
        ["DANGER_OBJECT", 1, '{"message":"object"}', (now + timedelta(minutes=4)).isoformat()],
    )
    reviewed = event_service.review_event(first, "confirm", "Operator verified")
    assert reviewed["review_status"] == "confirmed"
    assert reviewed["reviewed"] is True
    incidents = incident_service.list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["event_count"] == 2


def test_event_review_and_audit_commit_atomically(platform_db, monkeypatch):
    event_id = database.execute(
        "insert into events (event_type, severity, timestamp) values (?, ?, datetime('now'))",
        ["DANGER_OBJECT", 2],
    )

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(event_service, "record_action_in_connection", fail_audit)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        event_service.review_event(event_id, "confirm", "Operator verified")

    event = database.fetch_one("select review_status, reviewed_at from events where id=?", [event_id])
    assert event["review_status"] in (None, "open")
    assert event["reviewed_at"] is None
    assert database.fetch_one(
        "select id from platform_audit_log where action='event.review' and entity_id=?",
        [str(event_id)],
    ) is None


def test_security_analytics_excludes_attendance_observations(platform_db):
    database.execute(
        "insert into events (event_type, severity, details_json, timestamp) values (?, ?, ?, datetime('now'))",
        ["ATTENDANCE_CLOCKIN", 0, '{"message":"arrival"}'],
    )
    database.execute(
        "insert into events (event_type, severity, details_json, timestamp) values (?, ?, ?, datetime('now'))",
        ["ZONE_INTRUSION", 2, '{"message":"restricted entry"}'],
    )

    from backend.services.analytics_service import security

    result = security(days=1)
    assert result["securityObservationTotal"] == 1
    assert result["securityCategories"] == [{"name": "ZONE_INTRUSION", "value": 1}]


def test_command_idempotency(platform_db, tmp_path, monkeypatch):
    commands_path = tmp_path / "commands.json"
    results_path = tmp_path / "results.json"
    monkeypatch.setattr(command_service, "COMMANDS_PATH", commands_path)
    monkeypatch.setattr(command_service, "COMMAND_RESULTS_PATH", results_path)
    first = command_service.create_command("save_snapshot", {}, idempotency_key="demo-1")
    second = command_service.create_command("save_snapshot", {}, idempotency_key="demo-1")
    assert first["id"] == second["id"]


def test_outbox_is_idempotent_and_rejects_sensitive_payload(platform_db):
    first = outbox_service.enqueue(
        "attendance.clocked_in",
        {"person_id": 7, "date": "2026-09-18", "method": "automatic"},
        event_id="attendance-event-7",
        aggregate_type="attendance",
        aggregate_id=7,
    )
    second = outbox_service.enqueue(
        "attendance.clocked_in",
        {"person_id": 7, "date": "2026-09-18", "method": "automatic"},
        event_id="attendance-event-7",
        aggregate_type="attendance",
        aggregate_id=7,
    )
    assert first["id"] == second["id"]
    assert outbox_service.summary()["pending"] == 1
    with pytest.raises(outbox_service.OutboxSecurityError):
        outbox_service.enqueue("bad.event", {"embedding": [0.1, 0.2]})


def test_outbox_claim_ack_and_failure_are_scoped_and_durable(platform_db):
    outbox_service.enqueue("health.event", {"status": "camera_online"}, event_id="health-1")
    claimed = outbox_service.claim_pending(worker_id="worker-a")
    assert len(claimed) == 1
    assert claimed[0]["status"] == "processing"
    assert outbox_service.acknowledge("health-1", "worker-a") is True
    assert outbox_service.summary()["counts"]["sent"] == 1

    outbox_service.enqueue("health.event", {"status": "camera_degraded"}, event_id="health-2")
    claimed = outbox_service.claim_pending(worker_id="worker-b")
    assert len(claimed) == 1
    failed = outbox_service.fail("health-2", "temporary sync failure", "worker-b")
    assert failed["status"] == "failed"
    assert failed["attempts"] == 1
    assert outbox_service.summary()["pending"] == 1


def test_outbox_reclaims_abandoned_worker_leases(platform_db):
    outbox_service.enqueue("health.event", {"status": "abandoned"}, event_id="lease-1")
    assert outbox_service.claim_pending(worker_id="crashed-worker")
    database.execute(
        "update platform_outbox set locked_at='2000-01-01 00:00:00' where event_id='lease-1'"
    )
    assert outbox_service.recover_stale_claims(lease_seconds=300) == 1
    row = database.fetch_one(
        "select status, locked_by, last_error from platform_outbox where event_id='lease-1'"
    )
    assert dict(row) == {
        "status": "failed",
        "locked_by": None,
        "last_error": "worker lease expired",
    }
    assert outbox_service.claim_pending(worker_id="replacement-worker")[0]["event_id"] == "lease-1"


def test_outbox_dead_letter_requeue_and_sent_retention_are_controlled(platform_db):
    outbox_service.enqueue("health.event", {"status": "dead"}, event_id="dead-1")
    assert outbox_service.claim_pending(worker_id="worker-c")
    database.execute("update platform_outbox set attempts=7 where event_id='dead-1'")
    dead = outbox_service.fail("dead-1", "permanent failure", "worker-c")
    assert dead["status"] == "dead_letter"
    assert outbox_service.requeue_dead_letter("dead-1", actor_id="admin") is True
    assert database.fetch_one(
        "select status from platform_outbox where event_id='dead-1'"
    )["status"] == "pending"
    assert database.fetch_one(
        "select action from platform_audit_log where action='outbox.requeue'"
    )

    outbox_service.enqueue("health.event", {"status": "sent"}, event_id="sent-1")
    assert outbox_service.claim_pending(worker_id="worker-d")
    assert outbox_service.acknowledge("sent-1", "worker-d") is True
    database.execute(
        "update platform_outbox set sent_at='2020-01-01T00:00:00+00:00' where event_id='sent-1'"
    )
    preview = outbox_service.purge_sent(30)
    assert preview["eligible"] == 1
    assert preview["deleted"] == 0
    result = outbox_service.purge_sent(30, execute_delete=True, actor_id="admin")
    assert result["deleted"] == 1
