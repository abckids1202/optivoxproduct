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
    unresolved_decision = core.attendance_decision(1, active_roster=True)
    assert unresolved_decision["eligible"] is False
    assert unresolved_decision["roster_match"] is False

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


def test_configured_active_roster_is_authoritative_for_attendance_and_security_names():
    core = CorrelationCore(identity_confirmation_observations=1)
    core.set_active_roster(["Ada"])
    now = time.monotonic()
    core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Ada")],
        observed_at_monotonic=now,
        source_frame_id=10,
    )

    allowed = core.attendance_decision(1)
    assert allowed["eligible"] is True
    assert allowed["roster_match"] is True
    assert allowed["roster_validation"]["status"] == "ACTIVE"
    assert core.snapshot()["entities"][0]["roster_match"] is True

    core.set_active_roster([])
    denied = core.attendance_decision(1)
    assert denied["eligible"] is False
    assert denied["roster_match"] is False
    assert "person_not_active_in_roster" in denied["reason"]
    assert core.snapshot()["entities"][0]["roster_match"] is False

    security = core.correlate_security_events(
        [("ZONE_INTRUSION", "ID_1", 0.95, "Restricted zone", {
            "track_id": 1, "zone_id": "lab", "entity_type": "person",
        })],
        source_frame_id=11,
        observed_at_monotonic=now + 0.1,
    )
    event = security["security_events"][0]
    assert event["target"] == "UNKNOWN"
    assert event["metadata"]["roster_match"] is False
    assert event["metadata"]["roster_validation"]["status"] == "INACTIVE_OR_MISSING"


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
    decision = core.attendance_decision(1, active_roster=True)
    assert decision["person_name"] is None
    assert decision["eligible"] is False


def test_occluded_or_expired_identity_is_not_reported_as_fresh_evidence():
    core = CorrelationCore(
        identity_confirmation_observations=1,
        identity_stale_after_sec=1.0,
    )
    now = time.monotonic()
    core.update(
        tracked={1: (40, 40)},
        faces_info=[_face("Ada")],
        observed_at_monotonic=now,
    )

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[],
        observed_at_monotonic=now + 1.2,
    )
    identity = state["entities"][0]["identity"]
    assert identity["state"] == "OCCLUDED"
    assert identity["current_evidence_fresh"] is False
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


def test_derived_embeddings_do_not_change_threshold_calibration_basis():
    from main import _enrollment_basis_embeddings, _compute_intra_class_variance, VisionSystem

    original_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    original_b = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    derived = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    record = {
        "embeddings": [original_a, original_b, derived],
        "embedding_provenance": [
            {"kind": "source"}, {"kind": "source"}, {"kind": "augmented"},
        ],
    }

    basis = _enrollment_basis_embeddings(record)
    assert basis == [original_a, original_b]
    assert _compute_intra_class_variance(basis)[0] < _compute_intra_class_variance(record["embeddings"])[0]

    vision = VisionSystem.__new__(VisionSystem)
    vision.face_db = {"Ada": {**record, "threshold": 0.61, "source_fingerprints": ["sha"]}}
    vision.cfg = {"FACE_RECOG_THRESHOLD": 0.6}
    report = vision.enrollment_quality_report("Ada")
    assert report["threshold_basis"] == "original_evidence"
    assert report["threshold_basis_count"] == 2
    assert report["source_fingerprint_count"] == 1


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


def test_guided_enrollment_can_finish_after_minimum_samples_without_terminal_input():
    from main import VisionSystem

    vision = VisionSystem.__new__(VisionSystem)
    vision.cfg = {
        "FACE_RECOG_THRESHOLD": 0.6,
        "MAX_ENROLLMENT_EMBEDDINGS": 10,
        "PERFORMANCE": {"ENROLLMENT_DUPLICATE_SIMILARITY": 0.9995},
    }
    vision.face_db = {}
    vision._pending_web_enrollment = None
    vision._last_enrollment_metadata = {}
    vision.find_duplicate_identities = lambda embeddings, exclude_name=None: []
    vision._web_enrollment = {
        "active": True,
        "person_name": "Ada",
        "min_embeddings": 5,
        "max_embeddings": 10,
        "embeddings": [np.full(4, index + 1, dtype=np.float32) for index in range(5)],
        "last_score": 91.0,
        "quality_scores": [90.0, 91.0, 92.0, 90.0, 91.0],
        "quality_metrics": [{"score": 90.0}],
        "pose_yaws": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "rejected_samples": 2,
        "metadata": {"consent_confirmed": True},
        "replace_existing": False,
    }

    result = vision.finish_web_enrollment()

    assert result["stage"] == "completed"
    assert result["accepted"] == 5
    assert result["quality_metrics"]
    assert vision._web_enrollment is None
    assert vision._pending_web_enrollment["person_name"] == "Ada"
