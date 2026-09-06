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
    profile = people_service.get_person(person_id)
    assert profile["absence_history"][0]["reason"] == "Medical appointment"
    assert database.fetch_one("select actor_id from platform_audit_log where action='people.update'")["actor_id"] == "admin-test"


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
