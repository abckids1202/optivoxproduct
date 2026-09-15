import time

from core.correlation import CorrelationCore
from core.entities import EntityLifecycle
from core.security import SecuritySignalEngine


def _face(name="Ada", state="CONFIRMED", liveness="REAL"):
    return {
        "oid": 1,
        "bbox": (0, 0, 80, 80),
        "name": name,
        "confidence": 0.91,
        "current_observation_similarity": 0.91,
        "second_score": 0.45,
        "margin": 0.46,
        "identity_state": state,
        "quality_score": 94.0,
        "quality_ok": True,
        "liveness_status": liveness,
    }


def test_security_is_retargeted_from_entity_state_not_raw_track_label():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    core.update(
        tracked={1: (40, 40)},
        faces_info=[_face()],
        source_frame_id=17,
        observed_at_monotonic=now,
        observed_at_wallclock="2026-09-15T10:00:00+00:00",
    )
    result = core.correlate_security_events(
        [("ZONE_INTRUSION", "ID_1", 0.95, "Unauthorized", {"track_id": 1, "zone_id": "lab"})],
        source_frame_id=17,
        observed_at_monotonic=now,
    )
    assert result["security_event_tuples"][0][1] == "Ada"
    assert result["security_event_tuples"][0][4]["entity_id"] == "cam_0:entity:1"

    contradicted = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Bea")],
        source_frame_id=18,
        observed_at_monotonic=now + 0.1,
    )
    assert contradicted["entities"][0]["identity"]["state"] == "CONTRADICTED"
    result = core.correlate_security_events(
        [("ZONE_INTRUSION", "ID_1", 0.95, "Unauthorized", {"track_id": 1, "zone_id": "lab"})],
        source_frame_id=18,
        observed_at_monotonic=now + 0.1,
    )
    assert result["security_event_tuples"][0][1] == "UNKNOWN"


def test_security_engine_allows_confirmed_roster_identity_only_after_canonical_context():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "staff", "bounds": [0, 0, 100, 100], "allowed_names": ["ada"]}],
        }
    })
    engine.update({1: (150, 150)}, faces_info=[_face()], now=1.0)
    events = engine.update({1: (20, 20)}, faces_info=[_face()], now=2.0)
    assert [event[0] for event in events] == ["ZONE_ENTRY"]


def test_entity_provenance_and_occlusion_do_not_make_attendance_eligible():
    now = time.monotonic()
    core = CorrelationCore(identity_confirmation_observations=1)
    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face()],
        source_frame_id=42,
        observed_at_monotonic=now,
        observed_at_wallclock="2026-09-15T10:00:00+00:00",
    )
    entity = state["entities"][0]
    assert entity["last_source_frame_id"] == 42
    assert core.attendance_decision(1, expected_name="ada")["eligible"] is True

    state = core.update(tracked={}, faces_info=[], source_frame_id=43, observed_at_monotonic=now + 3.2)
    assert state["entities"][0]["lifecycle_state"] in {
        EntityLifecycle.STALE.value, EntityLifecycle.OCCLUDED.value,
    }
    assert core.attendance_decision(1)["eligible"] is False
