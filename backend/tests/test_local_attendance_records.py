from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

_test_config_dir = Path(__file__).resolve().parents[1] / ".pytest-temp" / "ultralytics"
_test_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_test_config_dir)
from main import EventDatabase, FAISSIndexer, VisionSystem, _utc_datetime


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
