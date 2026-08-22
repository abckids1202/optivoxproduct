import json
from datetime import datetime, timedelta

import pytest

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import attendance_service, command_service, event_service, incident_service, people_service


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


def test_command_idempotency(platform_db, tmp_path, monkeypatch):
    commands_path = tmp_path / "commands.json"
    results_path = tmp_path / "results.json"
    monkeypatch.setattr(command_service, "COMMANDS_PATH", commands_path)
    monkeypatch.setattr(command_service, "COMMAND_RESULTS_PATH", results_path)
    first = command_service.create_command("save_snapshot", {}, idempotency_key="demo-1")
    second = command_service.create_command("save_snapshot", {}, idempotency_key="demo-1")
    assert first["id"] == second["id"]
