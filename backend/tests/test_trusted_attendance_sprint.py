import time

import cv2
import numpy as np
import pytest

from core.correlation import CorrelationCore
from core.face_augmentation import generate_variants


def _face(name, state="CONFIRMED", liveness="REAL"):
    return {
        "oid": 1,
        "bbox": (0, 0, 80, 80),
        "name": name,
        "confidence": 0.90,
        "current_observation_similarity": 0.90,
        "identity_state": state,
        "quality_score": 92.0,
        "quality_ok": True,
        "liveness_status": liveness,
    }


def test_unknown_and_uncertain_entities_cannot_be_attendance_eligible():
    core = CorrelationCore(identity_confirmation_observations=3)
    now = time.monotonic()

    unresolved = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("UNKNOWN", "UNRESOLVED")],
        observed_at_monotonic=now,
    )
    assert unresolved["entities"][0]["attendance_eligibility"] is False
    assert core.attendance_decision(1, active_roster=True)["eligible"] is False

    uncertain = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Ada", "CANDIDATE", "UNCERTAIN")],
        observed_at_monotonic=now + 0.1,
    )
    assert uncertain["entities"][0]["attendance_eligibility"] is False


def test_confirmation_requires_consistent_observations_and_revalidation_after_occlusion():
    core = CorrelationCore(
        identity_confirmation_observations=3,
        face_visible_after_sec=1.0,
        quality_valid_after_sec=1.0,
        liveness_valid_after_sec=1.0,
    )
    now = time.monotonic()

    for index in range(3):
        state = core.update(
            tracked={1: (40, 40)},
            faces_info=[_face("Ada")],
            observed_at_monotonic=now + index * 0.1,
        )
    assert state["entities"][0]["identity"]["state"] == "CONFIRMED"
    assert state["entities"][0]["attendance_eligibility"] is True

    occluded = core.update(
        tracked={},
        faces_info=[],
        observed_at_monotonic=now + 3.2,
    )
    assert occluded["entities"][0]["identity"]["state"] == "OCCLUDED"
    assert occluded["entities"][0]["attendance_eligibility"] is False

    for index in range(3):
        state = core.update(
            tracked={1: (40, 40)},
            faces_info=[_face("Ada")],
            observed_at_monotonic=now + 3.3 + index * 0.1,
        )
        if index < 2:
            assert state["entities"][0]["identity"]["state"] == "CANDIDATE"
            assert state["entities"][0]["attendance_eligibility"] is False
    assert state["entities"][0]["identity"]["state"] == "CONFIRMED"


def test_contradictory_identity_downgrades_a_confirmed_entity():
    core = CorrelationCore(identity_confirmation_observations=1)
    now = time.monotonic()
    core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Ada")],
        observed_at_monotonic=now,
    )

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Bea")],
        observed_at_monotonic=now + 0.1,
    )
    identity = state["entities"][0]["identity"]
    assert identity["state"] == "CONTRADICTED"
    assert identity["contradiction_count"] == 1
    assert state["entities"][0]["attendance_eligibility"] is False


def test_augmentation_is_bounded_and_never_returns_an_exact_copy():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    variants = generate_variants(image, max_variants=3)

    assert len(variants) == 3
    assert len({label for label, _ in variants}) == 3
    assert all(not np.array_equal(image, variant) for _, variant in variants)
    assert all(variant.shape == image.shape for _, variant in variants)
    assert all(variant.dtype == image.dtype for _, variant in variants)
    assert cv2.mean(image) != cv2.mean(variants[0][1])


def test_failed_enrollment_persistence_restores_previous_identity_set():
    from main import VisionSystem

    vision = VisionSystem.__new__(VisionSystem)
    vision.cfg = {
        "FACE_RECOG_THRESHOLD": 0.6,
        "MAX_ENROLLMENT_EMBEDDINGS": 10,
        "PERFORMANCE": {"ENROLLMENT_DUPLICATE_SIMILARITY": 0.9995},
    }
    original = {"Ada": {"embeddings": [np.ones(4, dtype=np.float32)]}}
    vision.face_db = original
    vision._pending_web_enrollment = {
        "person_name": "Ada",
        "embeddings": [np.zeros(4, dtype=np.float32)],
        "quality_scores": [90.0],
        "pose_yaws": [0.0],
        "rejected_samples": 0,
        "duplicate_candidates": [],
        "replace_existing": True,
    }
    vision._rebuild_index = lambda: None
    vision.save_face_db = lambda: False

    with pytest.raises(RuntimeError):
        vision.commit_web_enrollment()
    assert vision.face_db is original
