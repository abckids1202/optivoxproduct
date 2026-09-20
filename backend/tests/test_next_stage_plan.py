import json
import time

import pytest

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import academic_service, attendance_service, auth_service, incident_service, people_service
from core.entities import EntityStateStore
from core.observations import Observation, ObservationType


@pytest.fixture()
def platform_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    yield


def add_person(name="Ada", metadata=None):
    return database.execute(
        "insert into people (name, role, metadata_json) values (?, 'Student', ?)",
        [name, json.dumps(metadata or {"class": "10A", "subjects": ["Math"]})],
    )


def test_identity_state_expires_after_occlusion_and_liveness_is_per_entity():
    store = EntityStateStore(identity_stale_after_sec=3.0, close_after_sec=20.0)
    now = time.monotonic()
    store.update_tracks({1: (10, 10), 2: (100, 100)}, now=now)
    store.attach(Observation(
        ObservationType.FACE_IDENTITY_RESULT, entity_track_id=1, observed_at_monotonic=now,
        ttl_ms=1000, value="Ada", confidence=.92,
        metadata={"identity_state": "CONFIRMED", "name": "Ada"},
    ))
    store.attach(Observation(
        ObservationType.LIVENESS_RESULT, entity_track_id=1, observed_at_monotonic=now,
        ttl_ms=1000, value="REAL",
    ))
    store.attach(Observation(
        ObservationType.LIVENESS_RESULT, entity_track_id=2, observed_at_monotonic=now,
        ttl_ms=1000, value="SUSPECT",
    ))

    at_occlusion = store.snapshot(now + 4.0)
    first = next(item for item in at_occlusion if item["track_id"] == 1)
    second = next(item for item in at_occlusion if item["track_id"] == 2)
    assert first["identity"]["state"] == "OCCLUDED"
    assert first["attendance_eligibility"] is False
    assert second["identity"]["state"] == "SPOOF_SUSPECT"

    at_expiry = store.snapshot(now + 10.0)
    first = next(item for item in at_expiry if item["track_id"] == 1)
    assert first["identity"]["state"] == "EXPIRED"


def test_profile_update_and_absence_are_auditable(platform_db):
    person_id = add_person()
    profile = people_service.update_person(
        person_id,
        {"className": "11B", "studentId": "S-001", "subjects": ["Physics", "Art"]},
        actor_id="admin-test",
    )
    assert profile["className"] == "11B"
    assert profile["studentId"] == "S-001"
    assert profile["subjects"] == ["Physics", "Art"]

    absence = attendance_service.record_absence(
        person_id, "2026-09-02", "excused", "Physics", "Medical appointment", actor_id="admin-test"
    )
    assert absence["official"] is True
    assert attendance_service.list_absences(person_id=person_id)[0]["status"] == "excused"
    assert database.fetch_one(
        "select count(*) as count from platform_outbox where event_type='attendance.absence_recorded'"
    )["count"] == 1
    profile = people_service.get_person(person_id)
    assert profile["absence_history"][0]["reason"] == "Medical appointment"
    assert database.fetch_one("select actor_id from platform_audit_log where action='people.update'")["actor_id"] == "admin-test"


def test_attendance_and_audit_commit_atomically(platform_db, monkeypatch):
    person_id = add_person()

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit storage unavailable")

    monkeypatch.setattr("backend.services.audit_service.append_audit_record", fail_audit)
    with pytest.raises(RuntimeError, match="audit storage unavailable"):
        attendance_service.clock_in(person_id, actor_id="operator-test")
    assert database.fetch_one(
        "select count(*) as count from attendance where person_id=?",
        [person_id],
    )["count"] == 0
    assert database.fetch_one("select count(*) as count from platform_outbox")["count"] == 0


def test_schedule_is_available_for_class_operations(platform_db):
    schedule = academic_service.create_schedule("10A", "Math", 0, "08:10", "09:00", 10)
    assert schedule["class_name"] == "10A"
    assert schedule["grace_minutes"] == 10
    assert academic_service.list_schedules(active_only=True)[0]["subject"] == "Math"


def test_incident_context_review_assignment_and_false_positive(platform_db):
    event_id = database.execute(
        "insert into events (event_type, severity, details_json, timestamp, entity_id, camera_id, location) values (?, ?, ?, ?, ?, ?, ?)",
        ["DANGER_OBJECT", 3, '{"message":"test"}', "2026-09-02T09:00:00", "cam_0:entity:1", "cam_0", "Gate"],
    )
    incidents = incident_service.list_incidents()
    assert incidents[0]["entityId"] == "cam_0:entity:1"
    assert incidents[0]["cameraId"] == "cam_0"
    incident_id = incidents[0]["id"]

    incident_service.assign_incident(incident_id, "security-lead", actor_id="operator-test")
    reviewed = incident_service.review_incident(incident_id, "false_positive", "Reviewed camera artifact", actor_id="operator-test")
    assert reviewed["status"] == "dismissed"
    assert reviewed["falsePositive"] is True
    assert reviewed["assignedTo"] == "security-lead"
    assert len(reviewed["review_actions"]) == 2


def test_incident_evidence_and_alert_are_linked_to_source_event(platform_db, tmp_path):
    evidence_path = tmp_path / "intrusion.jpg"
    evidence_path.write_bytes(b"test-evidence")
    event_id = database.execute(
        """insert into events
           (event_type, severity, details_json, timestamp, entity_id,
            presence_session_id, camera_id, location, snapshot_path, source_frame_id)
           values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ["ZONE_INTRUSION", 2,
         json.dumps({"security_metadata": {"zone_id": "lab", "track_id": 3}}),
         "2026-09-02T09:00:00", "cam_0:entity:3", 8, "cam_0", "Lab",
         str(evidence_path), 77],
    )
    database.execute(
        "insert into alert_log (channel, event_type, target, status, timestamp, source_event_id) values (?, ?, ?, ?, ?, ?)",
        ["webhook", "ZONE_INTRUSION", "ID_3", "delivered", "2026-09-02T09:00:01", event_id],
    )
    incident = incident_service.list_incidents()[0]
    detail = incident_service.get_incident(incident["id"])
    assert detail["zoneId"] == "lab"
    assert detail["presenceSessionId"] == 8
    assert detail["evidence"][0]["status"] == "available"
    assert detail["alerts"][0]["status"] == "delivered"
    assert detail["evidence_count"] == 1
    assert detail["alert_count"] == 1
    assert detail["alerts"][0]["delivered_at"] == "2026-09-02T09:00:01"


def _create_evidence_incident(tmp_path, monkeypatch):
    monkeypatch.setattr(incident_service, "SNAPSHOTS_DIR", tmp_path)
    evidence_path = tmp_path / "incident.jpg"
    evidence_path.write_bytes(b"trusted-evidence")
    event_id = database.execute(
        """insert into events
           (event_type, severity, details_json, timestamp, entity_id,
            camera_id, location, snapshot_path, source_frame_id)
           values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ["ZONE_INTRUSION", 2,
         json.dumps({"security_metadata": {"zone_id": "gate"}}),
         "2026-09-02T09:00:00", "cam_0:entity:9", "cam_0", "Gate",
         str(evidence_path), 901],
    )
    incident = incident_service.list_incidents()[0]
    assert incident_service.get_incident(incident["id"])["evidence"][0]["status"] == "available"
    return incident["id"], event_id, evidence_path


def test_incident_detail_revalidates_tampered_evidence_once(platform_db, tmp_path, monkeypatch):
    incident_id, event_id, evidence_path = _create_evidence_incident(tmp_path, monkeypatch)
    evidence_path.write_bytes(b"replacement-evidence")

    detail = incident_service.get_incident(incident_id)
    assert detail["evidence"][0]["status"] == "tampered"
    assert database.fetch_one(
        "select status from incident_evidence where incident_id=?", [incident_id]
    )["status"] == "tampered"
    assert database.fetch_one(
        "select count(*) as count from platform_audit_log where action='evidence.integrity_failure' and entity_id=?",
        [str(detail["evidence"][0]["id"])],
    )["count"] == 1

    # Re-opening the same incident must not create an unbounded audit storm.
    incident_service.get_incident(incident_id)
    assert database.fetch_one(
        "select count(*) as count from platform_audit_log where action='evidence.integrity_failure' and entity_id=?",
        [str(detail["evidence"][0]["id"])],
    )["count"] == 1
    assert database.fetch_one("select id from events where id=?", [event_id])["id"] == event_id


def test_incident_detail_marks_missing_and_unsafe_evidence(platform_db, tmp_path, monkeypatch):
    incident_id, _, evidence_path = _create_evidence_incident(tmp_path, monkeypatch)
    evidence_path.unlink()
    assert incident_service.get_incident(incident_id)["evidence"][0]["status"] == "missing"

    unsafe_path = tmp_path / ".." / ".." / ".." / ".." / "outside-evidence.jpg"
    database.execute(
        "update incident_evidence set path=? where incident_id=?",
        [str(unsafe_path), incident_id],
    )
    assert incident_service.get_incident(incident_id)["evidence"][0]["status"] == "invalid_path"
    assert database.fetch_one(
        "select count(*) as count from platform_audit_log where action='evidence.integrity_failure'"
    )["count"] == 1


def test_incident_grouping_keeps_same_correlation_id_separate_by_camera(platform_db):
    shared = "correlation-reused-by-fixture"
    for camera, entity in (("cam_0", "cam_0:entity:1"), ("cam_1", "cam_1:entity:1")):
        database.execute(
            """insert into events
               (event_type, severity, details_json, timestamp, entity_id,
                camera_id, location, correlation_id)
               values (?, ?, ?, ?, ?, ?, ?, ?)""",
            ["ZONE_INTRUSION", 2, "{}", "2026-09-02T09:00:00", entity,
             camera, "Gate", shared],
        )

    incidents = incident_service.list_incidents()

    assert len(incidents) == 2
    assert {item["cameraId"] for item in incidents} == {"cam_0", "cam_1"}


def test_password_sessions_store_only_hashes(platform_db):
    database.execute(
        "insert into platform_users (username, password_hash, role) values (?, ?, ?)",
        ["operator", auth_service.hash_password("correct horse"), "operator"],
    )
    result = auth_service.login("operator", "correct horse")
    assert result["user"]["role"] == "operator"
    assert result["access_token"]
    assert auth_service.resolve_session(result["access_token"])["username"] == "operator"
    stored = database.fetch_one("select password_hash from platform_users where username='operator'")
    assert stored["password_hash"] != "correct horse"
