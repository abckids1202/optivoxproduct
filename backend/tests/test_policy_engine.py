from __future__ import annotations

from datetime import datetime

from core.correlation import CorrelationCore
from core.policy_engine import PolicyEngine
from core.replay import run_replay
from core.security import SecuritySignalEngine
from backend.routes.system import _redact_policy_value


def test_policy_snapshot_is_stable_and_content_addressed():
    first = PolicyEngine({"ATTENDANCE": {"SCHOOL_DAYS": [0, 1, 2]}}).snapshot
    second = PolicyEngine({"ATTENDANCE": {"SCHOOL_DAYS": [0, 1, 2]}}).snapshot
    changed = PolicyEngine({"ATTENDANCE": {"SCHOOL_DAYS": [0, 1, 2, 3]}}).snapshot

    assert first.valid is True
    assert first.policy_id == second.policy_id
    assert first.policy_id != changed.policy_id
    assert first.metadata()["policy_version"] == first.policy_id


def test_policy_snapshot_rejects_invalid_calendar_and_thresholds():
    snapshot = PolicyEngine({
        "ATTENDANCE": {
            "SCHOOL_DAYS": [0, 0, 8],
            "HOLIDAYS": ["not-a-date"],
            "LATE_GRACE_MIN": -1,
        },
        "SECURITY": {"ZONES": "not-a-list"},
    }).snapshot

    assert snapshot.valid is False
    assert any("school_days" in issue for issue in snapshot.issues)
    assert any("holiday" in issue for issue in snapshot.issues)
    assert any("late_grace_min" in issue for issue in snapshot.issues)
    assert any("security.zones" in issue for issue in snapshot.issues)
    assert snapshot.attendance_school_day(datetime(2026, 9, 21)) is False


def test_policy_snapshot_rejects_invalid_security_geometry_and_thresholds():
    snapshot = PolicyEngine({
        "SECURITY": {
            "ZONES": [{"id": "lab", "shape": "circle", "bounds": [0, 0, 10, 10]}],
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": "not-a-number"},
            "PPE": {"MIN_OVERLAP": 2},
        }
    }).snapshot

    assert snapshot.valid is False
    assert any("security.zones[0].shape" in issue for issue in snapshot.issues)
    assert any("security.running.speed_threshold" in issue for issue in snapshot.issues)
    assert any("security.ppe.min_overlap" in issue for issue in snapshot.issues)


def test_security_signal_carries_policy_provenance():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "restricted", "bounds": [0, 0, 100, 100]}],
        }
    })
    engine.update({1: (150, 150)}, faces_info=[{"oid": 1}], now=1.0)
    events = engine.update({1: (20, 20)}, faces_info=[{"oid": 1}], now=2.0)

    assert events
    assert events[0][4]["policy_version"] == engine.policy_version


def test_replay_exposes_policy_identity():
    result = run_replay({
        "policy": {"identity_confirmation_observations": 1},
        "active_roster": ["Ada"],
        "frames": [{
            "time": 0,
            "tracks": [{"id": 1, "center": [40, 40]}],
            "faces": [{
                "oid": 1,
                "name": "Ada",
                "identity_state": "CONFIRMED",
                "quality_ok": True,
                "quality_score": 95,
                "liveness_status": "REAL",
                "confidence": 0.95,
                "current_observation_similarity": 0.95,
                "second_score": 0.2,
                "margin": 0.75,
            }],
        }],
    })

    assert result["policy_valid"] is True
    assert result["policy_version"]


def test_invalid_policy_fails_closed_for_attendance():
    core = CorrelationCore(
        identity_confirmation_observations=1,
        policy_valid=False,
        policy_issues=("attendance.school_days is invalid",),
    )
    core.update(
        tracked={1: (40, 40)},
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 80, 80),
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "quality_ok": True,
            "quality_score": 95,
            "liveness_status": "REAL",
            "confidence": 0.98,
            "current_observation_similarity": 0.98,
            "second_score": 0.2,
            "margin": 0.78,
        }],
    )

    decision = core.attendance_decision(1, active_roster=["Ada"])
    assert decision["eligible"] is False
    assert "policy_invalid" in decision["reason"]
    assert decision["policy_valid"] is False


def test_policy_history_redacts_sensitive_configuration():
    redacted = _redact_policy_value({
        "attendance": {"school_days": [0, 1]},
        "webhook_url": "https://example.invalid/secret",
        "smtp_pass": "do-not-return",
        "nested": [{"api_key": "hidden"}],
    })

    assert redacted["attendance"]["school_days"] == [0, 1]
    assert redacted["webhook_url"] == "[REDACTED]"
    assert redacted["smtp_pass"] == "[REDACTED]"
    assert redacted["nested"][0]["api_key"] == "[REDACTED]"
