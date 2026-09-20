import time
import json

from core.assistant_policy import (
    ASSISTANT_FORBIDDEN_ACTIONS,
    assistant_tool_allowed,
    assistant_tool_policy,
)
from core.correlation import CorrelationCore
from core.entities import EntityStateStore
from core.liveness import LivenessChallenge
from core.observations import Observation, ObservationType


def _face(name="Ada", state="CONFIRMED", liveness="REAL", reason="fresh"):
    return {
        "oid": 1,
        "bbox": (0, 0, 80, 80),
        "name": name,
        "confidence": 0.92,
        "current_observation_similarity": 0.92,
        "second_score": 0.40,
        "margin": 0.52,
        "identity_state": state,
        "quality_score": 94.0,
        "quality_ok": True,
        "liveness_status": liveness,
        "reason": reason,
    }


def test_cached_identity_is_displayable_but_cannot_authorize_attendance():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    core.update(tracked={1: (40, 40)}, faces_info=[_face()], observed_at_monotonic=now)
    assert core.attendance_decision(1, expected_name="Ada")["eligible"] is True

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(reason="stable_track_cache")],
        observed_at_monotonic=now + 0.1,
    )
    entity = state["entities"][0]
    assert entity["identity"]["confirmed_name"] == "Ada"
    assert entity["identity"]["current_evidence_fresh"] is False
    decision = core.attendance_decision(1, expected_name="Ada")
    assert decision["eligible"] is False
    assert "identity_evidence_cached_or_missing" in decision["reason"]

    correlated = core.correlate_security_events(
        [("ZONE_INTRUSION", "ID_1", 0.9, "Restricted zone", {"track_id": 1})],
        source_frame_id=2,
        observed_at_monotonic=now + 0.1,
    )
    assert correlated["security_event_tuples"][0][1] == "UNKNOWN"


def test_explicit_cached_only_flag_blocks_attendance_even_with_a_new_reason():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    core.update(tracked={1: (40, 40)}, faces_info=[_face()], observed_at_monotonic=now)

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(reason="diagnostic_continuity") | {"cached_only": True}],
        observed_at_monotonic=now + 0.1,
    )
    entity = state["entities"][0]
    assert entity["identity"]["current_evidence_fresh"] is False
    assert core.attendance_decision(1, expected_name="Ada")["eligible"] is False


def test_cached_only_evidence_flag_is_preserved_as_non_authoritative_metadata():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    result = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(reason="diagnostic_continuity") | {"cached_only": True}],
        observed_at_monotonic=now,
    )
    identity_event = core.entities.history.latest(
        1, ObservationType.FACE_IDENTITY_RESULT, now
    )
    assert identity_event is not None
    assert identity_event.metadata["cached_only"] is True
    assert result["entities"][0]["attendance_eligibility"] is False


def test_unknown_and_poor_quality_faces_remain_ineligible_without_danger_label():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(name="UNKNOWN", state="UNRESOLVED", liveness="UNCERTAIN")],
        observed_at_monotonic=now,
    )
    entity = state["entities"][0]
    assert entity["identity"]["state"] == "UNRESOLVED"
    assert entity["attendance_eligibility"] is False
    assert "UNKNOWN" not in entity["security_signals"]

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(name="UNKNOWN", state="UNRESOLVED", liveness="UNCERTAIN") | {
            "quality_ok": False,
            "quality_score": 20.0,
        }],
        observed_at_monotonic=now + 0.1,
    )
    gate = state["entities"][0]["attendance_decision"]
    assert gate["eligible"] is False
    assert "WAITING_FOR_GOOD_FACE" in gate["reason"]


def test_poor_quality_does_not_destroy_last_identity_but_waits_for_new_evidence():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    core.update(tracked={1: (40, 40)}, faces_info=[_face()], observed_at_monotonic=now)
    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face(reason="WAITING_FOR_GOOD_FACE") | {
            "quality_ok": False,
            "quality_score": 18.0,
        }],
        observed_at_monotonic=now + 0.1,
    )
    entity = state["entities"][0]
    assert entity["identity"]["confirmed_name"] == "Ada"
    assert entity["attendance_eligibility"] is False
    assert "WAITING_FOR_GOOD_FACE" in entity["attendance_decision"]["reason"]


def test_tracker_generation_requires_fresh_temporal_confirmation_after_switch():
    now = time.monotonic()
    core = CorrelationCore(
        identity_confirmation_observations=2,
        track_switch_distance=50,
    )
    core.update(tracked={1: (10, 10)}, faces_info=[_face(state="CANDIDATE")], observed_at_monotonic=now)
    state = core.update(
        tracked={1: (10, 10)},
        faces_info=[_face(state="CONFIRMED")],
        observed_at_monotonic=now + 0.1,
    )
    old_generation = state["entities"][0]["track_generation"]
    assert core.attendance_decision(1)["eligible"] is True

    state = core.update(
        tracked={1: (200, 200)},
        faces_info=[_face(state="CONFIRMED")],
        observed_at_monotonic=now + 1.0,
    )
    entity = state["entities"][0]
    assert entity["track_generation"] > old_generation
    assert entity["identity"]["confirmation_hits"] == 1
    assert core.attendance_decision(1)["eligible"] is False


def test_liveness_challenge_enforces_direction_and_records_outcomes():
    challenge = LivenessChallenge({
        "CENTER_MODE_CHALLENGE_TIMEOUT_SEC": 2,
        "CENTER_MODE_TURN_HOLD_SEC": 0.2,
        "CENTER_MODE_YAW_THRESHOLD": 0.12,
        "CENTER_MODE_NEUTRAL_THRESHOLD": 0.07,
        "CENTER_MODE_LEFT_YAW_SIGN": 1,
    })
    key = "cam:entity:1"
    challenge.update(key, 0.0, 0.0)
    assert challenge.update(key, 0.0, 0.3)["phase"] == "TURN_LEFT"
    assert "LEFT" in challenge.update(key, -0.2, 0.4)["message"]
    challenge.update(key, 0.2, 0.5)
    assert challenge.update(key, 0.2, 0.8)["phase"] == "TURN_RIGHT"
    challenge.update(key, -0.2, 0.9)
    assert challenge.update(key, -0.2, 1.2)["phase"] == "RETURN"
    challenge.update(key, 0.0, 1.3)
    result = challenge.update(key, 0.0, 1.6)
    assert result["status"] == "REAL"
    assert result["passed"] is True
    assert challenge.metrics_snapshot()["completions"] == 1
    assert challenge.state_for("cam:entity:2") is None

    challenge.record_evaluation("live_face", "REAL", True)
    challenge.record_evaluation("printed_photo", "SPOOF_SUSPECT", False)
    challenge.record_evaluation("phone_screen", "REAL", False)
    challenge.record_evaluation("lighting_changes", "UNCERTAIN", True)
    metrics = challenge.metrics_snapshot()
    assert metrics["true_accepts"] == 1
    assert metrics["true_rejects"] == 1
    assert metrics["false_accepts"] == 1
    assert metrics["false_rejects"] == 1
    assert metrics["false_accept_rate"] == 0.5
    assert metrics["false_reject_rate"] == 0.5
    assert metrics["scenarios"]["phone_screen"]["false_accept_rate"] == 1.0


def test_liveness_timeout_is_counted_and_entity_states_are_isolated():
    challenge = LivenessChallenge({"CENTER_MODE_CHALLENGE_TIMEOUT_SEC": 1})
    challenge.update("entity-a", 0.0, 0.0)
    timed_out = challenge.update("entity-a", 0.0, 2.0)
    assert timed_out["timed_out"] is True
    assert challenge.metrics_snapshot()["timeouts"] == 1
    assert challenge.update("entity-b", 0.0, 2.0)["attempt_number"] != timed_out["attempt_number"]

    store = EntityStateStore()
    now = time.monotonic()
    store.update_tracks({1: (10, 10), 2: (100, 100)}, now=now)
    store.attach(Observation(
        ObservationType.LIVENESS_RESULT,
        entity_track_id=1,
        observed_at_monotonic=now,
        value="REAL",
    ))
    store.attach(Observation(
        ObservationType.LIVENESS_RESULT,
        entity_track_id=2,
        observed_at_monotonic=now,
        value="SPOOF_SUSPECT",
    ))
    assert store.get(1).liveness.state == "REAL"
    assert store.get(2).liveness.state == "SPOOF_SUSPECT"


def test_anti_spoof_history_is_reset_when_track_generation_changes():
    from main import AntiSpoofDetector

    detector = AntiSpoofDetector({"ANTI_SPOOFING": {"ENABLED": True}})
    state, _, _, _ = detector._state_for_track(7)
    state["frames_seen"] = 9
    detector.bind_track_entity(7, "cam:entity:1", 1)
    state, _, _, _ = detector._state_for_track(7)
    assert state["frames_seen"] == 9
    state["frames_seen"] = 5
    detector.bind_track_entity(7, "cam:entity:2", 2)
    state, _, _, _ = detector._state_for_track(7)
    assert state["frames_seen"] == 0


def test_assistant_policy_is_read_only_and_blocks_sensitive_actions():
    assert assistant_tool_allowed("get_system_status") is True
    assert assistant_tool_allowed("clock_in") is False
    assert "delete_biometric" in ASSISTANT_FORBIDDEN_ACTIONS
    policy = assistant_tool_policy("change_permissions")
    assert policy == {
        "tool": "change_permissions",
        "allowed": False,
        "mode": "blocked",
        "requires_authorized_operator": True,
    }


def test_liveness_metrics_are_exposed_by_authenticated_live_state_service(tmp_path, monkeypatch):
    from backend.services import runtime_service

    metrics_path = tmp_path / "liveness_metrics.json"
    metrics_path.write_text(json.dumps({
        "status": "UNCERTAIN",
        "metrics": {"false_accepts": 0, "false_rejects": 1},
    }), encoding="utf-8")
    monkeypatch.setattr(runtime_service, "LIVENESS_METRICS_PATH", metrics_path)
    assert runtime_service.live_state()["liveness"]["metrics"]["false_rejects"] == 1
