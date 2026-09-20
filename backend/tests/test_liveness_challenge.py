from core.liveness import LivenessChallenge


def test_liveness_challenge_is_deterministic_and_entity_scoped():
    challenge = LivenessChallenge({
        "CENTER_MODE_CHALLENGE_TIMEOUT_SEC": 10,
        "CENTER_MODE_TURN_HOLD_SEC": 0.3,
        "CENTER_MODE_YAW_THRESHOLD": 0.12,
        "CENTER_MODE_NEUTRAL_THRESHOLD": 0.07,
    })
    key = "cam_0:entity:1"
    challenge.update(key, 0.0, 0.0)
    assert challenge.update(key, 0.0, 0.4)["phase"] == "TURN_LEFT"
    challenge.update(key, 0.2, 0.5)
    assert challenge.update(key, 0.2, 0.9)["phase"] == "TURN_RIGHT"
    challenge.update(key, -0.2, 1.0)
    assert challenge.update(key, -0.2, 1.4)["phase"] == "RETURN"
    challenge.update(key, 0.0, 1.5)
    assert challenge.update(key, 0.0, 1.9)["passed"] is True

    other = challenge.update("cam_0:entity:2", 0.0, 0.0)
    assert other["passed"] is False


def test_liveness_timeout_retries_are_persisted_per_entity_and_eventually_terminal():
    challenge = LivenessChallenge({
        "CENTER_MODE_CHALLENGE_TIMEOUT_SEC": 1,
        "CENTER_MODE_MAX_LIVENESS_ATTEMPTS": 2,
    })
    key = "cam_0:entity:timeout"

    challenge.update(key, 0.0, 0.0)
    first = challenge.update(key, 0.0, 2.0)
    assert first["timed_out"] is True
    assert first["terminal"] is False
    assert first["retry_count"] == 1
    assert challenge.state_for(key)["retry_count"] == 1

    second = challenge.update(key, 0.0, 4.0)
    assert second["timed_out"] is True
    assert second["terminal"] is True
    assert second["status"] == "SPOOF_SUSPECT"
    assert challenge.state_for(key)["failure_count"] == 2
