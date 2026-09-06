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
