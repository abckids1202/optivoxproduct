import time
import pytest

from core.correlation import CorrelationCore
from core.entities import EntityLifecycle, EntityStateStore
from core.observations import Observation, ObservationHistory, ObservationType


def test_observation_preserves_frame_provenance_and_expiry():
    now = time.monotonic()
    observation = Observation(
        ObservationType.FACE_QUALITY,
        camera_id="cam-test",
        source_frame_id=42,
        observed_at_monotonic=now,
        ttl_ms=100,
        producer="test",
        quality=87.0,
    )

    assert observation.source_frame_id == 42
    assert observation.camera_id == "cam-test"
    assert observation.age_ms(now + 0.05) == pytest.approx(50.0, abs=0.01)
    assert not observation.is_expired(now + 0.05)
    assert observation.is_expired(now + 0.101)


def test_observation_history_is_bounded_and_excludes_expired_values():
    history = ObservationHistory(max_per_type=2, max_entities=4)
    now = time.monotonic()
    for index in range(3):
        history.add(Observation(
            ObservationType.TRACK_MOTION,
            entity_track_id=7,
            observed_at_monotonic=now,
            ttl_ms=1000,
            value=index,
        ))

    assert [item.value for item in history.recent(7, ObservationType.TRACK_MOTION, now=now)] == [1, 2]
    assert history.stats()["observations"] == 2


def test_entity_state_keeps_identity_and_liveness_independent_per_track():
    store = EntityStateStore(max_entities=4)
    now = time.monotonic()
    store.update_tracks({1: (10, 10), 2: (100, 100)}, now=now)

    store.attach(Observation(
        ObservationType.FACE_IDENTITY_RESULT,
        entity_track_id=1,
        observed_at_monotonic=now,
        ttl_ms=1000,
        value="Matthew",
        confidence=0.86,
        metadata={"identity_state": "CONFIRMED", "name": "Matthew"},
    ))
    store.attach(Observation(
        ObservationType.LIVENESS_RESULT,
        entity_track_id=2,
        observed_at_monotonic=now,
        ttl_ms=1000,
        value="UNCERTAIN",
    ))

    first = store.get(1)
    second = store.get(2)
    assert first.identity.confirmed_name == "Matthew"
    assert first.liveness.state == "NOT_EVALUATED"
    assert second.identity.confirmed_name is None
    assert second.liveness.state == "UNCERTAIN"


def test_entity_lifecycle_closes_after_track_disappears():
    store = EntityStateStore(occluded_after_updates=2, close_after_sec=1.0)
    now = time.monotonic()
    store.update_tracks({4: (20, 20)}, now=now)
    store.update_tracks({}, now=now + 0.1)
    assert store.get(4).lifecycle_state == EntityLifecycle.STALE
    store.update_tracks({}, now=now + 0.2)
    assert store.get(4).lifecycle_state == EntityLifecycle.OCCLUDED
    store.update_tracks({}, now=now + 1.2)
    assert store.get(4) is None


def test_tracker_id_is_scoped_to_camera_and_cannot_carry_identity_across_streams():
    store = EntityStateStore()
    now = time.monotonic()
    store.update_tracks({1: (20, 20)}, camera_id="cam-a", now=now)
    first = store.get(1)
    first_id = first.entity_id
    first_generation = first.track_generation
    store.consume_transitions()

    store.update_tracks({1: (20, 20)}, camera_id="cam-b", now=now + 0.1)
    second = store.get(1)
    transitions = store.consume_transitions()

    assert second.entity_id != first_id
    assert second.camera_id == "cam-b"
    assert second.track_generation == first_generation + 1
    assert any(item["reason"] == "camera_changed" for item in transitions)


def test_expired_pose_is_not_reported_as_current_entity_state():
    store = EntityStateStore()
    now = time.monotonic()
    store.update_tracks({8: (40, 40)}, now=now)
    store.attach(Observation(
        ObservationType.POSE_STATE,
        entity_track_id=8,
        observed_at_monotonic=now,
        ttl_ms=50,
        value="FALL",
    ))
    assert store.snapshot(now + 0.02)[0]["pose_state"] == "FALL"
    assert store.snapshot(now + 0.10)[0]["pose_state"] is None


def test_correlation_core_attaches_faces_and_keeps_multi_person_pose_explicit():
    core = CorrelationCore(max_entities=8, max_observations_per_type=4)
    state = core.update(
        tracked={1: (10, 10), 2: (110, 110)},
        source_frame_id=99,
        camera_id="cam-test",
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 20, 20),
            "name": "Matthew",
            "confidence": 0.86,
            "current_observation_similarity": 0.86,
            "identity_state": "CONFIRMED",
            "quality_score": 91,
            "quality_ok": True,
            "liveness_status": "REAL",
            "yaw": 0.02,
        }],
        pose_result={"pose": object(), "is_fallen": True, "hands_raised": False},
        object_detections=[{
            "class_name": "backpack",
            "confidence": 0.81,
            "bbox": (50, 50, 80, 80),
        }],
    )

    entity = next(item for item in state["entities"] if item["track_id"] == 1)
    other = next(item for item in state["entities"] if item["track_id"] == 2)
    assert state["frame_id"] == 99
    assert entity["entity_id"].startswith("cam-test:entity:")
    assert entity["identity"]["confirmed_name"] == "Matthew"
    assert entity["face"]["quality"] == 91
    assert other["identity"]["confirmed_name"] is None
    assert state["stats"]["global_observations"] >= 1
    assert state["stats"]["observations_created"] >= 8
    object_observation = core.global_history.latest(None, "OBJECT_DETECTED")
    assert object_observation is not None
    assert object_observation.model_name == "YOLO"
    assert object_observation.metadata["event_type"] == "OBJECT_DETECTED"
    assert object_observation.metadata["execution_mode"] == "active"


def test_entity_attendance_requires_confirmed_identity_and_real_liveness():
    core = CorrelationCore(max_entities=2)
    state = core.update(
        tracked={1: (20, 20)},
        source_frame_id=7,
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 40, 40),
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "quality_score": 90,
            "quality_ok": True,
            "liveness_status": "UNCERTAIN",
        }],
    )
    assert state["entities"][0]["attendance_eligibility"] is False

    state = core.update(
        tracked={1: (20, 20)},
        source_frame_id=8,
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 40, 40),
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "quality_score": 90,
            "quality_ok": True,
            "liveness_status": "REAL",
        }],
    )
    assert state["entities"][0]["attendance_eligibility"] is True


def test_cached_module_provenance_is_not_rewritten_as_current_frame():
    core = CorrelationCore(max_entities=4, max_observations_per_type=4)
    now = time.monotonic()
    core.update(
        tracked={1: (10, 10)},
        source_frame_id=100,
        observed_at_monotonic=now,
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 20, 20),
            "name": "Matthew",
            "identity_state": "CANDIDATE",
        }],
        provenance={"faces": {"frame_id": 95, "monotonic": now - 0.4}},
    )

    identity = core.entities.history.latest(1, "FACE_IDENTITY_RESULT", now=now)
    assert identity is not None
    assert identity.source_frame_id == 95
    assert identity.age_ms(now) == pytest.approx(400.0, abs=0.01)


def test_entity_provenance_does_not_move_backwards_for_delayed_observations():
    store = EntityStateStore()
    now = time.monotonic()
    store.update_tracks({3: (20, 20)}, now=now)
    store.attach(Observation(
        ObservationType.FACE_DETECTED,
        entity_track_id=3,
        observed_at_monotonic=now + 1.0,
        observed_at_wallclock="2026-09-17T10:00:01+00:00",
        source_frame_id=20,
        bbox=(0, 0, 40, 40),
    ))
    store.attach(Observation(
        ObservationType.FACE_QUALITY,
        entity_track_id=3,
        observed_at_monotonic=now + 0.2,
        observed_at_wallclock="2026-09-17T10:00:00.200000+00:00",
        source_frame_id=19,
        quality=60.0,
    ))

    entity = store.get(3)
    assert entity.last_seen_monotonic == pytest.approx(now + 1.0)
    assert entity.last_observed_wallclock == "2026-09-17T10:00:01+00:00"
    assert entity.last_source_frame_id == 20


def test_security_events_are_enriched_from_correlated_entity_state():
    core = CorrelationCore(max_entities=4, identity_confirmation_observations=1)
    state = core.update(
        tracked={1: (20, 20)},
        source_frame_id=21,
        faces_info=[{
            "oid": 1,
            "bbox": (0, 0, 40, 40),
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "quality_ok": True,
            "quality_score": 90,
            "liveness_status": "REAL",
        }],
        security_events=[(
            "ZONE_INTRUSION", "ID_1", 0.95, "Restricted zone entry",
            {"track_id": 1, "zone_id": "lab"},
        )],
    )

    event = state["security_events"][0]
    assert event["target"] == "Ada"
    assert event["metadata"]["entity_id"].startswith("cam_0:entity:")
    assert event["metadata"]["identity_state"] == "CONFIRMED"
    assert event["metadata"]["liveness_state"] == "REAL"
    assert state["security_event_tuples"][0][4]["track_generation"] == 1


def test_late_security_events_use_the_same_entity_authority():
    core = CorrelationCore(max_entities=4, identity_confirmation_observations=1)
    now = time.monotonic()
    core.update(
        tracked={7: (20, 20)},
        source_frame_id=30,
        observed_at_monotonic=now,
        faces_info=[{
            "oid": 7,
            "bbox": (0, 0, 40, 40),
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "quality_ok": True,
            "quality_score": 95,
            "liveness_status": "REAL",
        }],
    )

    result = core.correlate_security_events(
        [(
            "SPOOF_DETECTED", "UNKNOWN", 0.9, "Repeated spoof suspicion",
            {"track_id": 7, "entity_type": "person"},
        ), (
            "DANGEROUS_OBJECT", "knife", 0.88, "Object confirmation",
            {"entity_type": "object", "object_class": "knife"},
        )],
        source_frame_id=31,
        camera_id="cam_0",
        observed_at_monotonic=now + 0.1,
    )

    spoof, object_event = result["security_events"]
    assert spoof["metadata"]["entity_id"].startswith("cam_0:entity:")
    assert spoof["metadata"]["entity_type"] == "person"
    assert spoof["metadata"]["track_generation"] == 1
    assert spoof["metadata"]["identity_state"] == "CONFIRMED"
    assert object_event["metadata"]["entity_id"] is None
    assert object_event["metadata"]["entity_type"] == "object"


def test_attendance_decision_rejects_stale_liveness():
    core = CorrelationCore(
        max_entities=2,
        identity_confirmation_observations=1,
        liveness_valid_after_sec=0.5,
    )
    now = time.monotonic()
    face = {
        "oid": 1,
        "bbox": (0, 0, 40, 40),
        "name": "Ada",
        "identity_state": "CONFIRMED",
        "quality_ok": True,
        "quality_score": 90,
        "liveness_status": "REAL",
    }
    core.update(tracked={1: (20, 20)}, faces_info=[face], observed_at_monotonic=now)
    assert core.attendance_decision(1)["eligible"] is True

    core.update(
        tracked={1: (20, 20)},
        faces_info=[{**face, "liveness_status": "NOT_EVALUATED"}],
        observed_at_monotonic=now + 1.0,
    )
    decision = core.attendance_decision(1)
    assert decision["eligible"] is False
    assert "entity_attendance_not_eligible" in decision["reason"]


def test_simultaneous_duplicate_identity_keeps_one_authoritative_entity():
    core = CorrelationCore(max_entities=4, identity_confirmation_observations=1)
    state = core.update(
        tracked={1: (20, 20), 2: (120, 120)},
        source_frame_id=50,
        faces_info=[
            {
                "oid": 1,
                "bbox": (0, 0, 40, 40),
                "name": "Ada",
                "identity_state": "CONFIRMED",
                "current_observation_similarity": 0.91,
                "quality_ok": True,
                "quality_score": 94,
                "liveness_status": "REAL",
            },
            {
                "oid": 2,
                "bbox": (100, 100, 140, 140),
                "name": "Ada",
                "identity_state": "CONFIRMED",
                "current_observation_similarity": 0.84,
                "quality_ok": True,
                "quality_score": 90,
                "liveness_status": "REAL",
            },
        ],
    )

    by_track = {item["track_id"]: item for item in state["entities"]}
    assert by_track[1]["identity"]["state"] == "CONFIRMED"
    assert by_track[1]["attendance_eligibility"] is True
    assert by_track[2]["identity"]["state"] == "CONTRADICTED"
    assert by_track[2]["attendance_eligibility"] is False
    assert len(state["identity_conflicts"]) == 1
    assert state["identity_conflicts"][0]["winner_track_id"] == 1


def test_identity_collision_is_deterministic_on_equal_scores():
    core = CorrelationCore(max_entities=4, identity_confirmation_observations=1)
    state = core.update(
        tracked={2: (20, 20), 7: (120, 120)},
        source_frame_id=51,
        faces_info=[
            {"oid": 2, "bbox": (0, 0, 40, 40), "name": "Ada", "identity_state": "CONFIRMED",
             "current_observation_similarity": 0.88, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
            {"oid": 7, "bbox": (100, 100, 140, 140), "name": "Ada", "identity_state": "CONFIRMED",
             "current_observation_similarity": 0.88, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
        ],
    )
    by_track = {item["track_id"]: item for item in state["entities"]}
    assert by_track[2]["identity"]["state"] == "CONFIRMED"
    assert by_track[7]["identity"]["state"] == "CONTRADICTED"


def test_identity_conflict_is_emitted_as_reviewable_non_danger_observation():
    core = CorrelationCore(max_entities=4, identity_confirmation_observations=1)
    face_one = {
        "oid": 1, "bbox": (0, 0, 40, 40), "name": "Ada",
        "identity_state": "CONFIRMED", "current_observation_similarity": 0.95,
        "quality_ok": True, "quality_score": 94, "liveness_status": "REAL",
    }
    first = core.update(
        tracked={1: (20, 20)}, faces_info=[face_one], source_frame_id=1)
    assert first["security_event_tuples"] == []

    result = core.update(
        tracked={1: (20, 20), 2: (120, 120)},
        faces_info=[face_one, {**face_one, "oid": 2,
                              "bbox": (100, 100, 140, 140),
                              "current_observation_similarity": 0.70}],
        source_frame_id=2,
    )
    conflict = next(
        event for event in result["security_event_tuples"]
        if event[0] == "IDENTITY_CONFLICT"
    )
    assert conflict[1] == "UNKNOWN"
    assert conflict[4]["entity_type"] == "person"
    assert conflict[4]["entity_id"] == "cam_0:entity:2"
    assert conflict[4]["winner_entity_id"] == "cam_0:entity:1"
    assert conflict[4]["correlation_id"]
    assert result["entities"][1]["attendance_eligibility"] is False
