import json

import pytest

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import attendance_service, operational_service


@pytest.fixture()
def platform_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    yield


def add_person(name: str) -> int:
    return database.execute(
        "insert into people (name, role, metadata_json) values (?, ?, ?)",
        [name, "Student", json.dumps({"class": "10A"})],
    )


def test_presence_session_and_evidence_are_traceable(platform_db):
    person_id = add_person("Ada")
    session_id = database.execute(
        """
        insert into presence_sessions
            (entity_id, track_id, person_id, label, identity_state,
             liveness_status, camera_id, started_at, last_seen_at,
             confidence, first_frame_id, last_frame_id)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["cam_0:entity-1", 4, person_id, "Ada", "CONFIRMED", "REAL",
         "cam_0", "2026-08-29T09:00:00+00:00", "2026-08-29T09:00:03+00:00",
         0.91, 100, 103],
    )
    evidence_id = database.execute(
        """
        insert into recognition_evidence
            (entity_id, presence_session_id, track_id, person_id,
             candidate_name, decision, similarity, quality_score,
             quality_ok, liveness_status, identity_state, reason,
             source_frame_id, observed_at, details_json)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["cam_0:entity-1", session_id, 4, person_id, "Ada", "confirmed",
         0.91, 88.0, 1, "REAL", "CONFIRMED", "temporal_confirmation",
         103, "2026-08-29T09:00:03+00:00", json.dumps({"margin": 0.14})],
    )

    result = operational_service.get_presence_session(session_id)

    assert result["entityId"] == "cam_0:entity-1"
    assert result["personId"] == person_id
    assert result["identityState"] == "CONFIRMED"
    assert result["evidenceCount"] == 1
    assert result["evidence"][0]["id"] == evidence_id
    assert result["evidence"][0]["sourceFrameId"] == 103
    assert result["evidence"][0]["details"] == {"margin": 0.14}


def test_unknown_presence_is_not_attendance(platform_db):
    database.execute(
        """
        insert into presence_sessions
            (entity_id, label, identity_state, camera_id, started_at, last_seen_at)
        values (?, 'UNKNOWN', 'UNRESOLVED', 'cam_0', ?, ?)
        """,
        ["cam_0:unknown-1", "2026-08-29T09:00:00+00:00", "2026-08-29T09:00:03+00:00"],
    )
    database.execute(
        """
        insert into recognition_evidence
            (entity_id, candidate_name, decision, quality_ok, identity_state,
             observed_at, reason)
        values (?, 'UNKNOWN', 'unresolved', 1, 'UNRESOLVED', ?, 'no_confirmed_candidate')
        """,
        ["cam_0:unknown-1", "2026-08-29T09:00:03+00:00"],
    )

    summary = operational_service.summary()
    attendance = attendance_service.attendance_summary()

    assert summary["activeUnresolvedEntities"] == 1
    assert summary["activeConfirmedEntities"] == 0
    assert summary["confirmedEvidence"] == 0
    assert attendance["present"] == 0
    assert database.fetch_one("select count(*) as c from attendance")["c"] == 0


def test_operational_summary_distinguishes_rejected_evidence(platform_db):
    database.execute(
        """
        insert into recognition_evidence
            (entity_id, candidate_name, decision, quality_ok, identity_state, observed_at)
        values (?, ?, ?, ?, ?, ?)
        """,
        ["cam_0:spoof-1", "SPOOF", "spoof_or_uncertain", 0, "UNRESOLVED", "2026-08-29T09:00:00+00:00"],
    )

    summary = operational_service.summary()

    assert summary["recognitionEvidence"] == 1
    assert summary["confirmedEvidence"] == 0
    assert summary["rejectedEvidence"] == 1


def test_operations_api_exposes_correlated_counts_and_validates_limits(platform_db):
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    summary = client.get("/api/operations/summary")
    invalid = client.get("/api/operations/presence?limit=0")

    assert summary.status_code == 200
    assert summary.json()["activePresenceSessions"] == 0
    assert invalid.status_code == 422


def test_liveness_challenge_api_exposes_safe_entity_trace(platform_db):
    database.execute(
        """
        insert into liveness_challenges
            (entity_id, track_id, track_generation, challenge_state, phase,
             liveness_status, started_at, updated_at, attempt_number,
             source_frame_id, failure_reason, metrics_json)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "cam_0:entity-1", 4, 2, "PASSED", "RETURN_FORWARD", "REAL",
            "2026-08-29T09:00:00+00:00", "2026-08-29T09:00:04+00:00", 1,
            104, None, json.dumps({"forward_frames": 4, "quality": 88, "embedding": "must-not-leak"}),
        ],
    )

    result = operational_service.list_liveness_challenges()

    assert result[0]["entityId"] == "cam_0:entity-1"
    assert result[0]["challengeState"] == "PASSED"
    assert result[0]["trackGeneration"] == 2
    assert result[0]["metrics"] == {"forward_frames": 4, "quality": 88}
    assert "embedding" not in result[0]["metrics"]

    from fastapi.testclient import TestClient

    from backend.main import app

    response = TestClient(app).get("/api/operations/liveness-challenges?limit=1")
    assert response.status_code == 200
    assert response.json()[0]["livenessStatus"] == "REAL"
