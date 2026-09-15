from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

_test_config_dir = Path(__file__).resolve().parents[1] / ".pytest-temp" / "ultralytics"
_test_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_test_config_dir)
from main import AttendanceManager, CONFIG, EventDatabase, FAISSIndexer, VisionSystem, _utc_datetime


def _database(tmp_path):
    database = EventDatabase(str(tmp_path / "security.db"))
    database.setup_database()
    return database


def test_presence_session_keeps_confirmed_identity_during_non_identity_updates(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        session_id = database.upsert_presence_session(
            "cam_0:entity:1", track_id=3, person_id=person_id,
            label="Ada", identity_state="CONFIRMED", liveness_status="REAL",
        )
        database.upsert_presence_session(
            "cam_0:entity:1", track_id=3, person_id=None,
            label="UNKNOWN", identity_state="OCCLUDED", liveness_status="NOT_EVALUATED",
        )
        resumed_occlusion = database.upsert_presence_session(
            "cam_0:entity:1", track_id=3, person_id=None,
            label="UNKNOWN", identity_state="OCCLUDED", liveness_status="NOT_EVALUATED",
            visible=False,
        )
        assert resumed_occlusion == session_id
        row = database._fetchone(
            "select * from presence_sessions where id=?", (session_id,))
        assert row["person_id"] == person_id
        assert row["label"] == "Ada"
        assert row["identity_state"] == "OCCLUDED"

        closed = database.close_stale_presence_sessions(
            timeout_sec=0.1, now=_utc_datetime() + timedelta(seconds=1))
        assert closed == 1
        closed_row = database._fetchone(
            "select status, closed_reason from presence_sessions where id=?",
            (session_id,),
        )
        assert closed_row["status"] == "closed"
        assert closed_row["closed_reason"] == "timeout"
    finally:
        database.conn.close()


def test_automatic_clockout_uses_last_valid_presence(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        clocked_in = database.attendance_clock_in(person_id)
        assert "clocked_in_at" in clocked_in
        old_seen = (_utc_datetime() - timedelta(minutes=16)).isoformat()
        with database.lock:
            database.conn.execute(
                "update attendance set last_seen_at=? where person_id=? and date=?",
                (old_seen, person_id, database._fetchone(
                    "select date from attendance where person_id=?",
                    (person_id,))["date"]),
            )
            database.conn.commit()

        closed = database.close_stale_attendance(
            timeout_min=15, now=_utc_datetime())
        assert len(closed) == 1
        row = database._fetchone(
            "select clock_out, clock_out_source, last_seen_at from attendance where person_id=?",
            (person_id,),
        )
        assert row["clock_out"] == old_seen
        assert row["clock_out_source"] == "automatic_timeout"
        assert row["last_seen_at"] == old_seen
    finally:
        database.conn.close()


def test_automatic_attendance_skips_configured_non_school_day(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Weekend Student")
        monkeypatch.setitem(CONFIG["ATTENDANCE"], "SCHOOL_DAYS", [6])
        attendance = AttendanceManager(database, CONFIG)
        attendance.handle_recognition(
            "Weekend Student", confidence=0.95, identity_state="CONFIRMED",
            liveness_status="REAL", attendance_eligible=True, quality_ok=True,
            presence_session_id="cam_0:entity:1",
        )
        assert database._fetchone(
            "select id from attendance where person_id=? and date=?",
            (person_id, database._fetchone("select date(?) as date", (_utc_datetime(),))["date"]),
        ) is None
    finally:
        database.conn.close()


def test_rollover_closes_presence_sessions_with_explicit_reason(tmp_path):
    database = _database(tmp_path)
    try:
        database.upsert_presence_session("cam_0:entity:rollover", track_id=4)
        assert database.close_open_presence_sessions("day_rollover") == 1
        row = database._fetchone(
            "select status, closed_reason from presence_sessions where entity_id=?",
            ("cam_0:entity:rollover",),
        )
        assert row["status"] == "closed"
        assert row["closed_reason"] == "day_rollover"
    finally:
        database.conn.close()


def test_attendance_decision_audit_is_idempotent_and_keeps_rejections(tmp_path):
    database = _database(tmp_path)
    person_id = database.upsert_person("Ada")
    first = database.record_attendance_decision(
        decision_key="cam_0:entity:1:44:rejected",
        decision="rejected",
        reason="identity_unresolved",
        entity_id="cam_0:entity:1",
        person_id=None,
        identity_state="UNRESOLVED",
        liveness_status="NOT_EVALUATED",
        source_frame_id=44,
        observed_at="2026-09-15T10:00:00+00:00",
    )
    second = database.record_attendance_decision(
        decision_key="cam_0:entity:1:44:rejected",
        decision="rejected",
        reason="should_not_duplicate",
        entity_id="cam_0:entity:1",
        source_frame_id=44,
        observed_at="2026-09-15T10:00:00+00:00",
    )
    assert first == second
    row = database._fetchone(
        "select decision, reason, person_id from attendance_decisions where id=?",
        (first,),
    )
    assert row["decision"] == "rejected"
    assert row["reason"] == "identity_unresolved"
    assert row["person_id"] is None
    database.conn.close()


def test_face_matching_uses_indexed_samples_and_records_margin():
    indexer = FAISSIndexer(dim=3)
    indexer.add_embeddings(
        ["Ada", "Ada", "Ada", "Bea"],
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.98, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [0.6, 0.6, 0.6, 0.6],
    )
    matcher = SimpleNamespace(
        face_db={"Ada": {"embeddings": [1, 2, 3]}, "Bea": {"embeddings": [4]}},
        face_indexer=indexer,
        cfg={"FACE_RECOG_ACCEPT_THRESHOLD": 0.62, "FACE_RECOG_MIN_MARGIN": 0.08},
        _last_recognition_details={},
    )

    name, confidence, reason = VisionSystem.recognize_face(
        matcher, [1.0, 0.0, 0.0])

    assert name == "Ada"
    assert confidence > 0.9
    assert "margin=" in reason
    assert matcher._last_recognition_details["second_score"] == 0.0
