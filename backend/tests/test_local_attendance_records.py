from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sqlite3
import threading
import pytest
from schema_migrations import verify_audit_chain

_test_config_dir = Path(__file__).resolve().parents[1] / ".pytest-temp" / "ultralytics"
_test_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_test_config_dir)
from main import (
    AttendanceManager,
    CONFIG,
    EventDatabase,
    FAISSIndexer,
    VisionSystem,
    _sync_runtime_incidents,
    _process_runtime_side_effects,
    _local_datetime,
    _today_iso,
    _utc_datetime,
)


def _database(tmp_path):
    database = EventDatabase(str(tmp_path / "security.db"))
    database.setup_database()
    return database


def test_canonical_database_transaction_rolls_back_and_reports_health(tmp_path):
    database = _database(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with database.transaction(immediate=True) as con:
                con.execute("create table transaction_probe(value text)")
                con.execute("insert into transaction_probe values ('aborted')")
                raise RuntimeError("abort")
        assert database._fetchone(
            "select name from sqlite_master where type='table' and name='transaction_probe'"
        ) is None
        health = database.health_report()
        assert health["status"] == "healthy"
        assert health["quick_check"] == "ok"
        assert health["foreign_key_violations"] == 0
        assert health["audit"]["ok"] is True
        assert health["outbox"]["status"] == "healthy"
        assert health["migration_states"][0]["schema_name"] == "edge"
        assert health["migration_states"][0]["version"] == 2
    finally:
        database.conn.close()


def test_canonical_runtime_uses_strict_durability_and_platform_contract(tmp_path, monkeypatch):
    import main as runtime

    monkeypatch.setattr(runtime, "SQLITE_SYNCHRONOUS", "FULL")
    monkeypatch.setattr(runtime, "SQLITE_SECURE_DELETE", "ON")
    database = _database(tmp_path)
    try:
        assert database.conn.execute("pragma synchronous").fetchone()[0] == 2
        assert database.conn.execute("pragma secure_delete").fetchone()[0] == 1
        assert database.conn.execute(
            "select 1 from sqlite_master where type='table' and name='platform_users'"
        ).fetchone() is not None
        assert database.conn.execute(
            "select 1 from sqlite_master where type='index' and name='idx_presence_one_open_per_entity'"
        ).fetchone() is not None
    finally:
        database.conn.close()


def test_canonical_records_persist_deployment_scope(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Scoped Student")
        event_id = database.log_event("CAMERA_HEALTH", person_id=person_id, event_uid="scope-event")
        session_id = database.upsert_presence_session(
            "cam_0:entity:scope", track_id=1, person_id=person_id,
            label="Scoped Student", identity_state="CONFIRMED", liveness_status="REAL",
        )
        evidence_id = database.record_recognition_evidence(
            "cam_0:entity:scope", presence_session_id=session_id,
            person_id=person_id, candidate_name="Scoped Student",
            decision="confirmed", quality_ok=True,
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        decision_id = database.record_attendance_decision(
            "scope-decision", "rejected", reason="test", person_id=person_id,
            entity_id="cam_0:entity:scope", presence_session_id=session_id,
            recognition_evidence_id=evidence_id,
        )
        for table, row_id in (
            ("events", event_id), ("presence_sessions", session_id),
            ("recognition_evidence", evidence_id), ("attendance_decisions", decision_id),
        ):
            row = database._fetchone(
                f"select organization_id, site_id, device_id from {table} where id=?",
                (row_id,),
            )
            assert row["organization_id"] == database.organization_id
            assert row["site_id"] == database.site_id
            assert row["device_id"] == database.device_id
    finally:
        database.conn.close()


def test_canonical_attendance_writes_transactional_outbox(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Outbox Student")
        clocked_in = database.attendance_clock_in(person_id, decision_source="manual")
        assert clocked_in["clocked_in_at"]
        queued = database._fetchone(
            "select event_type, status, organization_id, site_id, device_id from platform_outbox"
        )
        assert queued["event_type"] == "attendance.clocked_in"
        assert queued["status"] == "pending"
        assert queued["organization_id"] == database.organization_id

        database.attendance_clock_out(person_id, source="manual")
        assert database._fetchone(
            "select count(*) as count from platform_outbox where event_type='attendance.clocked_out'"
        )["count"] == 1
    finally:
        database.conn.close()


def test_event_persistence_preserves_source_occurrence_time(tmp_path):
    database = _database(tmp_path)
    try:
        event_id = database.log_event(
            event_type="ZONE_ENTRY",
            camera_id="cam_0",
            source_frame_id=42,
            correlation_id="corr-source-time",
            occurred_at="2026-09-17T08:15:30.125+00:00",
            event_uid="event-source-time",
        )
        event = database._fetchone("select * from events where id=?", (event_id,))
        assert event["timestamp"] == "2026-09-17T08:15:30.125+00:00"
        assert event["source_frame_id"] == 42
        assert event["correlation_id"] == "corr-source-time"
        daily = database._fetchone(
            "select date from event_stats_daily where event_type=?",
            ("ZONE_ENTRY",),
        )
        hourly = database._fetchone(
            "select hour from event_stats_hourly where event_type=?",
            ("ZONE_ENTRY",),
        )
        assert daily["date"] == "2026-09-17"
        assert hourly["hour"] == "2026-09-17 08:00"
    finally:
        database.conn.close()


def test_edge_audit_records_are_hash_chained_and_tamper_evident(tmp_path):
    database = _database(tmp_path)
    try:
        database.log_audit("runtime.action", "cam_0", {"safe": True})
        database.log_audit("runtime.action", "cam_0", {"step": 2})
        assert database.audit_integrity()["ok"]
        database.conn.execute("drop trigger trg_auditlog_append_only_update")
        database.conn.execute(
            "update audit_log set details_json=? where action=?",
            ('{"tampered":true}', "runtime.action"),
        )
        database.conn.commit()
        report = verify_audit_chain(database.conn, "audit_log")
        assert report["ok"] is False
        assert report["reason"] == "record_hash_mismatch"
    finally:
        database.conn.close()


def test_runtime_security_event_materializes_traceable_incident_offline(tmp_path):
    database = _database(tmp_path)
    try:
        event_id = database.log_event(
            event_type="ZONE_INTRUSION",
            confidence=0.93,
            details={"message": "restricted zone entry", "security_metadata": {"zone_id": "vault"}},
            camera_id="cam_0",
            location="Gate",
            severity=2,
            entity_id="cam_0:entity:4",
            presence_session_id=12,
            source_frame_id=88,
            correlation_id="corr-zone-4",
            event_uid="event-zone-4",
        )
        database.conn.execute(
            "insert into alert_log (channel, event_type, target, status, timestamp, source_event_id) values (?, ?, ?, ?, ?, ?)",
            ("webhook", "ZONE_INTRUSION", "vault", "delivered", "2026-09-17T10:00:01", event_id),
        )
        database.conn.commit()

        assert _sync_runtime_incidents(database, [event_id], str(tmp_path / "snapshots")) == 1
        incident = database._fetchone("select * from incidents")
        assert incident["category"] == "Security"
        assert incident["entity_id"] == "cam_0:entity:4"
        assert incident["presence_session_id"] == 12
        assert database._fetchone("select * from incident_events")["event_id"] == event_id
        alert = database._fetchone("select * from incident_alerts")
        assert alert["channel"] == "webhook"
        assert alert["status"] == "delivered"

        # Replaying the same operations task is safe and does not duplicate
        # the incident, event link, or alert link.
        assert _sync_runtime_incidents(database, [event_id], str(tmp_path / "snapshots")) == 0
        assert database._fetchone("select count(*) as count from incidents")["count"] == 1
        assert database._fetchone("select count(*) as count from incident_events")["count"] == 1
        assert database._fetchone("select count(*) as count from incident_alerts")["count"] == 1
    finally:
        database.conn.close()


def test_malformed_event_severity_falls_back_without_dropping_persistence(tmp_path):
    database = _database(tmp_path)
    try:
        class _Alerts:
            def check_and_alert(self, **kwargs):
                return None

        result = _process_runtime_side_effects(
            {
                "frame_id": 7,
                "completed_at": _utc_datetime().isoformat(),
                "faces_info": [],
                "events": [(
                    "ZONE_INTRUSION", "UNKNOWN", 0.8, "restricted entry",
                    {"severity": "NaN", "entity_type": "object"},
                )],
                    "correlation": {
                        "enabled": True,
                        "decision_boundary": "CORRELATION_CORE",
                    },
            },
            database,
            SimpleNamespace(),
            _Alerts(),
            tmp_path,
            "cam_0",
            "test",
            threading.Lock(),
            lambda: True,
        )

        event = database._fetchone("select event_type, severity from events")
        assert result["events_persisted"] == 1
        assert event["event_type"] == "ZONE_INTRUSION"
        assert event["severity"] == 2
    finally:
        database.conn.close()


def test_uncorrelated_runtime_task_cannot_persist_security_or_attendance_decisions(tmp_path):
    database = _database(tmp_path)
    try:
        result = _process_runtime_side_effects(
            {
                "frame_id": 9,
                "completed_at": _utc_datetime().isoformat(),
                "faces_info": [{
                    "entity_id": "cam_0:entity:raw",
                    "oid": 3,
                    "name": "Ada",
                    "evidence_due": True,
                    "identity_state": "CONFIRMED",
                    "liveness_status": "REAL",
                    "quality_ok": True,
                }],
                "events": [("ZONE_INTRUSION", "Ada", 0.9, "raw event", {})],
                "correlation": {},
            },
            database,
            SimpleNamespace(),
            SimpleNamespace(check_and_alert=lambda **kwargs: None),
            tmp_path,
            "cam_0",
            "test",
            threading.Lock(),
            lambda: False,
        )

        assert result["events_persisted"] == 0
        assert result["attendance_decisions"] == 0
        assert result["uncorrelated_decisions_rejected"] == 2
        assert database._fetchone("select count(*) as count from events")["count"] == 0
        assert database._fetchone(
            "select count(*) as count from recognition_evidence"
        )["count"] == 0
    finally:
        database.conn.close()


def test_inactive_confirmed_label_is_stored_as_roster_rejected_evidence(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        database.set_person_active(person_id, False)
        _process_runtime_side_effects(
            {
                "frame_id": 17,
                "completed_at": _utc_datetime().isoformat(),
                "faces_info": [{
                    "entity_id": "cam_0:entity:inactive",
                    "oid": 4,
                    "name": "Ada",
                    "evidence_due": True,
                    "identity_state": "CONFIRMED",
                    "liveness_status": "REAL",
                    "quality_ok": True,
                    "quality_score": 92,
                    "current_observation_similarity": 0.94,
                }],
                "events": [],
                    "correlation": {
                        "enabled": True,
                        "decision_boundary": "CORRELATION_CORE",
                        "entities": [{
                        "entity_id": "cam_0:entity:inactive",
                        "track_id": 4,
                        "track_generation": 1,
                        "lifecycle_state": "ACTIVE",
                        "roster_match": False,
                        "roster_validation": {
                            "configured": True,
                            "match": False,
                            "status": "INACTIVE_OR_MISSING",
                        },
                        "attendance_eligibility": False,
                        "identity": {
                            "state": "CONFIRMED",
                            "confirmed_name": "Ada",
                            "best_score": 0.94,
                            "current_evidence_fresh": True,
                        },
                        "liveness": {"state": "REAL"},
                    }],
                },
            },
            database,
            SimpleNamespace(),
            SimpleNamespace(check_and_alert=lambda **kwargs: None),
            tmp_path,
            "cam_0",
            "test",
            threading.Lock(),
            lambda: True,
        )
        evidence = database._fetchone(
            "select candidate_name, person_id, decision, details_json from recognition_evidence")
        assert evidence["candidate_name"] == "Ada"
        assert evidence["person_id"] is None
        assert evidence["decision"] == "roster_rejected"
        assert "INACTIVE_OR_MISSING" in evidence["details_json"]
        decision = database._fetchone(
            "select decision, reason from attendance_decisions")
        assert decision["decision"] == "rejected"
        assert "person_not_active_in_roster" in decision["reason"]
    finally:
        database.conn.close()


def test_vehicle_event_persists_its_entity_without_person_context(tmp_path):
    database = _database(tmp_path)
    try:
        result = _process_runtime_side_effects(
            {
                "frame_id": 8,
                "completed_at": _utc_datetime().isoformat(),
                "faces_info": [],
                "events": [(
                    "VEHICLE_ENTERED", "VEHICLE_1", 0.9, "vehicle entered",
                    {"entity_id": "cam_0:vehicle:1", "entity_type": "vehicle",
                     "track_id": 1},
                )],
                    "correlation": {
                        "enabled": True,
                        "decision_boundary": "CORRELATION_CORE",
                    },
            },
            database,
            SimpleNamespace(),
            SimpleNamespace(check_and_alert=lambda **kwargs: None),
            tmp_path,
            "cam_0",
            "test",
            threading.Lock(),
            lambda: True,
        )
        event = database._fetchone(
            "select entity_id, presence_session_id, person_id from events "
            "where event_type='VEHICLE_ENTERED'"
        )
        assert result["events_persisted"] == 1
        assert event["entity_id"] == "cam_0:vehicle:1"
        assert event["presence_session_id"] is None
        assert event["person_id"] is None
    finally:
        database.conn.close()


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


def test_occluded_only_update_does_not_fabricate_a_new_presence_session(tmp_path):
    database = _database(tmp_path)
    try:
        session_id = database.upsert_presence_session(
            "cam_0:entity:occluded-only", track_id=7,
            identity_state="OCCLUDED", visible=False,
        )
        assert session_id is None
        assert database._fetchone(
            "select count(*) as count from presence_sessions",
        )["count"] == 0
    finally:
        database.conn.close()


def test_automatic_clockout_uses_last_valid_presence(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        clocked_in = database.attendance_clock_in(person_id)
        assert "clocked_in_at" in clocked_in
        base_time = _utc_datetime() - timedelta(minutes=30)
        old_seen = (base_time + timedelta(minutes=14)).isoformat()
        with database.lock:
            database.conn.execute(
                "update attendance set clock_in=?, last_seen_at=? where person_id=? and date=?",
                (base_time.isoformat(), old_seen, person_id, database._fetchone(
                    "select date from attendance where person_id=?",
                    (person_id,))["date"]),
            )
            database.conn.commit()

        closed = database.close_stale_attendance(
            timeout_min=15, now=base_time + timedelta(minutes=30))
        assert len(closed) == 1
        assert closed[0]["camera_id"] is None
        assert closed[0]["source_frame_id"] is None
        row = database._fetchone(
            "select clock_out, clock_out_source, last_seen_at from attendance where person_id=?",
            (person_id,),
        )
        assert row["clock_out"] == old_seen
        assert row["clock_out_source"] == "automatic_timeout"
        assert row["last_seen_at"] == old_seen
    finally:
        database.conn.close()


def test_presence_and_attendance_last_seen_are_monotonic_under_out_of_order_tasks(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        session_id = database.upsert_presence_session(
            "cam_0:entity:ordered", track_id=3, person_id=person_id,
            label="Ada", identity_state="CONFIRMED", liveness_status="REAL",
            observed_at="2026-09-17T10:00:00+00:00",
        )
        database.upsert_presence_session(
            "cam_0:entity:ordered", track_id=3, person_id=person_id,
            label="Ada", identity_state="CONFIRMED", liveness_status="REAL",
            observed_at="2026-09-17T09:59:00+00:00",
        )
        session = database._fetchone(
            "select last_seen_at from presence_sessions where id=?", (session_id,))
        assert session["last_seen_at"] == "2026-09-17T10:00:00+00:00"

        database.attendance_clock_in(
            person_id, presence_session_id=session_id,
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        database.touch_attendance_presence(
            person_id, "2026-09-17T09:58:00+00:00", session_id)
        attendance = database._fetchone(
            "select last_seen_at from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
        )
        # The clock-in itself is newer than the deliberately stale task.
        assert attendance["last_seen_at"] != "2026-09-17T09:58:00+00:00"
    finally:
        database.conn.close()


def test_cached_visible_entity_refreshes_open_attendance_presence(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Ada")
        session_id = database.upsert_presence_session(
            "cam_0:entity:cached", track_id=7, person_id=person_id,
            label="Ada", identity_state="CONFIRMED", liveness_status="REAL",
        )
        database.attendance_clock_in(
            person_id, presence_session_id=session_id,
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        old_seen = (_utc_datetime() - timedelta(minutes=16)).isoformat()
        with database.lock:
            database.conn.execute(
                "update attendance set last_seen_at=? where person_id=? and date=?",
                (old_seen, person_id, database._fetchone(
                    "select date from attendance where person_id=?",
                    (person_id,),
                )["date"]),
            )
            database.conn.commit()

        class _Attendance:
            def reconcile(self, force=False):
                return []

        class _Alerts:
            def check_and_alert(self, **kwargs):
                return None

        _process_runtime_side_effects(
            {
                "frame_id": 20,
                "completed_at": _utc_datetime().isoformat(),
                "faces_info": [],  # no fresh evidence: this models a cache hit
                "events": [],
                    "correlation": {
                        "enabled": True,
                        "decision_boundary": "CORRELATION_CORE",
                        "entities": [{
                        "entity_id": "cam_0:entity:cached",
                        "track_id": 7,
                        "track_generation": 1,
                        "lifecycle_state": "ACTIVE",
                        "attendance_eligibility": True,
                        "identity": {
                            "state": "CONFIRMED",
                            "confirmed_name": "Ada",
                            "best_score": 0.96,
                            "current_evidence_fresh": True,
                        },
                        "liveness": {"state": "REAL"},
                    }],
                },
            },
            database,
            _Attendance(),
            _Alerts(),
            tmp_path,
            "cam_0",
            "test",
            threading.Lock(),
            lambda: True,
        )

        row = database._fetchone(
            "select last_seen_at from attendance where person_id=? and date=?",
            (person_id, database._fetchone(
                "select date from attendance where person_id=?",
                (person_id,),
            )["date"]),
        )
        assert row["last_seen_at"] != old_seen
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


def test_invalid_matching_schedule_fails_closed_instead_of_using_global_policy(tmp_path):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Invalid Schedule", metadata={"class": "10A"})
        database.conn.execute(
            "insert into attendance_schedules "
            "(class_name, subject, weekday, start_time, end_time, grace_minutes, organization_id, site_id, device_id) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("10A", "Math", _local_datetime().weekday(), "25:99", "10:00", 5,
             database.organization_id, database.site_id, database.device_id),
        )
        database.conn.commit()
        result = database.attendance_clock_in(
            person_id, decision_source="automatic", camera_id="cam_0",
        )
        assert result["schedule_invalid"] is True
        assert database._fetchone(
            "select id from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
        ) is None
    finally:
        database.conn.close()


def test_schedule_resolution_does_not_cross_site_scope(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        monkeypatch.setattr("main._is_school_day", lambda _: True)
        person_id = database.upsert_person("Scoped Schedule", metadata={"class": "10A"})
        database.conn.execute(
            """insert into attendance_schedules
               (class_name, subject, weekday, start_time, end_time, grace_minutes,
                organization_id, site_id, device_id)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("10A", "Math", _local_datetime().weekday(), "25:99", "10:00", 5,
             database.organization_id, "different-site", database.device_id),
        )
        database.conn.commit()

        result = database.attendance_clock_in(
            person_id, decision_source="automatic", camera_id="cam_0",
        )

        assert result.get("schedule_invalid") is not True
        assert "provenance_required" in result
    finally:
        database.conn.close()


def test_non_authoritative_recognition_cannot_create_attendance(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        monkeypatch.setattr("main._is_school_day", lambda _: True)
        person_id = database.upsert_person("Legacy Label")
        attendance = AttendanceManager(database, CONFIG)
        for _ in range(20):
            attendance.handle_recognition(
                "Legacy Label", confidence=0.99, identity_state="CONFIRMED",
                liveness_status="REAL", attendance_eligible=True,
                quality_ok=True, authoritative=False,
            )
        assert database._fetchone(
            "select id from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
        ) is None
        rejection = database._fetchone(
            "select action from audit_log where action=? order by id desc limit 1",
            ("ATTENDANCE_REJECTED_NON_AUTHORITATIVE",),
        )
        assert rejection is not None
    finally:
        database.conn.close()


def test_automatic_database_clockin_requires_correlated_provenance(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        monkeypatch.setattr("main._is_school_day", lambda _: True)
        person_id = database.upsert_person("Provenance Student")

        rejected = database.attendance_clock_in(
            person_id,
            decision_source="automatic_correlated",
            identity_state="CONFIRMED",
            liveness_status="REAL",
        )

        assert rejected["provenance_required"] is True
        assert database._fetchone(
            "select id from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
        ) is None

        session_id = database.upsert_presence_session(
            "cam_0:entity:provenance", track_id=9,
            person_id=person_id, label="Provenance Student",
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        evidence_id = database.record_recognition_evidence(
            "cam_0:entity:provenance", presence_session_id=session_id,
            track_id=9, person_id=person_id, candidate_name="Provenance Student",
            decision="confirmed", quality_ok=True,
            liveness_status="REAL", identity_state="CONFIRMED",
        )
        accepted = database.attendance_clock_in(
            person_id, presence_session_id=session_id,
            recognition_evidence_id=evidence_id,
            decision_source="automatic_correlated",
            identity_state="CONFIRMED", liveness_status="REAL",
        )

        assert "clocked_in_at" in accepted
        row = database._fetchone(
            "select policy_version from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
        )
        assert row["policy_version"] == database.policy_engine.snapshot.policy_id
        snapshot = database._fetchone(
            """select declared_version, document_json, issues_json, valid
               from policy_snapshots where policy_version=?
                 and organization_id=? and site_id=? and device_id=?""",
            (
                database.policy_engine.snapshot.policy_id,
                database.organization_id,
                database.site_id,
                database.device_id,
            ),
        )
        assert snapshot["declared_version"] == database.policy_engine.snapshot.declared_version
        assert json.loads(snapshot["document_json"])["attendance"]["school_days"]
        assert json.loads(snapshot["issues_json"]) == []
        assert snapshot["valid"] == 1
        with pytest.raises(sqlite3.IntegrityError):
            database.conn.execute(
                "update policy_snapshots set document_json='{}' where policy_version=?",
                (database.policy_engine.snapshot.policy_id,),
            )
    finally:
        database.conn.close()


def test_automatic_database_clockin_rejects_mismatched_or_closed_provenance(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        monkeypatch.setattr("main._is_school_day", lambda _: True)
        person_id = database.upsert_person("Bound Student")
        other_id = database.upsert_person("Other Student")
        session_id = database.upsert_presence_session(
            "cam_0:entity:bound", track_id=4, person_id=person_id,
            label="Bound Student", identity_state="CONFIRMED", liveness_status="REAL",
        )
        evidence_id = database.record_recognition_evidence(
            "cam_0:entity:bound", presence_session_id=session_id,
            track_id=4, person_id=person_id, candidate_name="Bound Student",
            decision="confirmed", quality_ok=True,
            liveness_status="REAL", identity_state="CONFIRMED",
        )

        mismatched = database.attendance_clock_in(
            other_id, presence_session_id=session_id,
            recognition_evidence_id=evidence_id,
            decision_source="automatic_correlated",
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        assert mismatched["provenance_invalid"] is True

        database.close_presence_session("cam_0:entity:bound", reason="timeout")
        closed = database.attendance_clock_in(
            person_id, presence_session_id=session_id,
            recognition_evidence_id=evidence_id,
            decision_source="center_attendance",
            identity_state="CONFIRMED", liveness_status="REAL",
        )
        assert closed["provenance_invalid"] is True
        assert database._fetchone(
            "select id from attendance where person_id=? and date=?",
            (person_id, _today_iso()),
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


def test_day_rollover_emits_traceable_automatic_clockout_event(tmp_path, monkeypatch):
    database = _database(tmp_path)
    try:
        person_id = database.upsert_person("Rollover Student")
        old_date = "2026-09-17"
        last_seen = "2026-09-17T14:10:00+00:00"
        database.conn.execute(
            "insert into attendance "
            "(person_id, date, clock_in, camera_id, location, source_frame_id, "
            "presence_session_id, last_seen_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
            (person_id, old_date, "2026-09-17T08:00:00+00:00", "cam_gate",
             "Main gate", 321, None, last_seen),
        )
        database.conn.commit()

        monkeypatch.setattr("main._today_iso", lambda: "2026-09-18")
        attendance = AttendanceManager(database, CONFIG)
        attendance._date = old_date
        attendance._maybe_rollover()

        event = database._fetchone(
            "select details_json, camera_id, location, source_frame_id from events "
            "where event_uid=?",
            (f"attendance:clock_out:{person_id}:{old_date}:day_rollover",),
        )
        assert event is not None
        assert event["camera_id"] == "cam_gate"
        assert event["location"] == "Main gate"
        assert event["source_frame_id"] == 321
        assert json.loads(event["details_json"])["source"] == "day_rollover"
    finally:
        database.conn.close()


def test_entity_transition_closes_presence_at_last_valid_seen_time(tmp_path):
    database = _database(tmp_path)
    try:
        last_seen = "2026-09-17T10:00:00+00:00"
        entity_id = "cam_0:entity:transition"
        database.upsert_presence_session(
            entity_id, track_id=4, observed_at=last_seen)
        assert database.close_presence_session(entity_id, reason="tracker_switch") is True
        row = database._fetchone(
            "select status, ended_at, closed_reason from presence_sessions where entity_id=?",
            (entity_id,),
        )
        assert row["status"] == "closed"
        assert row["ended_at"] == last_seen
        assert row["closed_reason"] == "tracker_switch"
    finally:
        database.conn.close()


def test_track_generation_change_starts_new_presence_session(tmp_path):
    database = _database(tmp_path)
    try:
        first_person = database.upsert_person("Ada")
        second_person = database.upsert_person("Bea")
        first = database.upsert_presence_session(
            "cam_0:entity:reused", track_id=4, person_id=first_person,
            label="Ada", identity_state="CONFIRMED", track_generation=1,
        )
        second = database.upsert_presence_session(
            "cam_0:entity:reused", track_id=4, person_id=second_person,
            label="Bea", identity_state="CONFIRMED", track_generation=2,
        )

        assert second != first
        old = database._fetchone("select status, closed_reason, person_id from presence_sessions where id=?", (first,))
        current = database._fetchone("select status, track_generation, person_id from presence_sessions where id=?", (second,))
        assert old["status"] == "closed"
        assert old["closed_reason"] == "track_generation_changed"
        assert old["person_id"] == first_person
        assert current["status"] == "active"
        assert current["track_generation"] == 2
        assert current["person_id"] == second_person
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


def test_liveness_challenge_is_persisted_without_biometric_material(tmp_path):
    database = _database(tmp_path)
    try:
        challenge_id = database.start_liveness_challenge(
            "cam_0:entity:liveness", track_id=5, track_generation=2,
            started_at="2026-09-17T10:00:00+00:00", source_frame_id=12,
        )
        assert challenge_id is not None
        assert database.update_liveness_challenge(
            challenge_id, phase="TURN_LEFT", liveness_status="UNCERTAIN",
            updated_at="2026-09-17T10:00:01+00:00", source_frame_id=13,
            metrics={"progress": 0.35},
        ) is True
        assert database.update_liveness_challenge(
            challenge_id, phase="PASSED", liveness_status="REAL",
            challenge_state="PASSED", updated_at="2026-09-17T10:00:04+00:00",
            completed_at="2026-09-17T10:00:04+00:00", source_frame_id=16,
        ) is True
        row = database._fetchone("select * from liveness_challenges where id=?", (challenge_id,))
        assert row["challenge_state"] == "PASSED"
        assert row["phase"] == "PASSED"
        assert row["track_generation"] == 2
        assert "embedding" not in (row["metrics_json"] or "").lower()
    finally:
        database.conn.close()


def test_runtime_event_stores_evidence_checksum_at_capture(tmp_path):
    database = _database(tmp_path)
    try:
        evidence = tmp_path / "event.jpg"
        evidence.write_bytes(b"deterministic evidence")
        event_id = database.log_event(
            "ZONE_INTRUSION",
            snapshot_path=str(evidence),
            correlation_id="corr_test",
        )
        row = database._fetchone(
            "select evidence_checksum, correlation_id from events where id=?",
            (event_id,),
        )
        assert row["correlation_id"] == "corr_test"
        assert row["evidence_checksum"] == hashlib.sha256(
            evidence.read_bytes()).hexdigest()
    finally:
        database.conn.close()


def test_correlated_event_uid_makes_runtime_event_persistence_idempotent(tmp_path):
    database = _database(tmp_path)
    try:
        first = database.log_event(
            "ZONE_INTRUSION", confidence=0.9,
            details={"message": "restricted entry"},
            correlation_id="corr-1", event_uid="evt-stable-1",
        )
        second = database.log_event(
            "ZONE_INTRUSION", confidence=0.2,
            details={"message": "retry should not duplicate"},
            correlation_id="corr-1", event_uid="evt-stable-1",
        )
        assert second == first
        assert database._fetchone(
            "select count(*) as count from events where event_uid=?",
            ("evt-stable-1",),
        )["count"] == 1
        assert database._fetchone(
            "select count from event_stats_daily where event_type=?",
            ("ZONE_INTRUSION",),
        )["count"] == 1
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
