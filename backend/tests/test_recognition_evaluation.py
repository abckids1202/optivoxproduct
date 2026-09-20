import numpy as np
import pytest

from core.recognition_evaluation import (
    RecognitionEvaluationError,
    compare_embedding_galleries,
    evaluate_embedding_gallery,
)


def _gallery():
    return [
        {"sample_id": "ada-1", "identity": "Ada", "kind": "original", "embedding": [1, 0, 0]},
        {"sample_id": "bo-1", "identity": "Bo", "kind": "original", "embedding": [0, 1, 0]},
        {"sample_id": "ada-aug-1", "identity": "Ada", "kind": "augmented",
         "derived_from": "ada-1", "embedding": [0.98, 0.2, 0]},
    ]


def test_embedding_evaluation_keeps_unknown_people_out_of_attendance_identity():
    result = evaluate_embedding_gallery(
        _gallery(),
        [
            {"sample_id": "ada-probe", "identity": "Ada", "embedding": [0.99, 0.1, 0]},
            {"sample_id": "unknown-probe", "identity": "UNKNOWN", "embedding": [0, 0, 1]},
        ],
        threshold=0.8,
        min_margin=0.1,
    )

    assert result["status"] == "MEASURED"
    assert result["false_accepts"] == 0
    assert result["unknown_rejection_accuracy"] == 1.0
    assert all("embedding" not in row for row in result["rows"])


def test_augmented_gallery_comparison_rejects_false_accept_increase():
    probes = [
        {"identity": "Ada", "embedding": [0.99, 0.1, 0]},
        {"identity": "UNKNOWN", "embedding": [0.8, 0.6, 0]},
    ]
    result = compare_embedding_galleries(
        _gallery()[:2], _gallery(), probes, threshold=0.85, min_margin=0.05
    )

    assert result["recommendation"] == "REJECT_AUGMENTATION_FALSE_ACCEPT_INCREASE"
    assert result["delta"]["false_accept_rate"] > 0


def test_invalid_or_unprovenanced_embedding_inputs_fail_closed():
    with pytest.raises(RecognitionEvaluationError, match="augmented sample requires"):
        evaluate_embedding_gallery(
            [{"sample_id": "aug", "identity": "Ada", "kind": "augmented", "embedding": [1, 0]}],
            [{"identity": "Ada", "embedding": [1, 0]}],
        )

    with pytest.raises(RecognitionEvaluationError, match="finite"):
        evaluate_embedding_gallery(
            [{"sample_id": "ada", "identity": "Ada", "embedding": [np.nan, 0]}],
            [{"identity": "Ada", "embedding": [1, 0]}],
        )


def test_probe_source_or_exact_embedding_overlap_is_rejected():
    with pytest.raises(RecognitionEvaluationError, match="overlaps gallery source"):
        evaluate_embedding_gallery(
            [{"sample_id": "ada", "identity": "Ada", "source_id": "session-1", "embedding": [1, 0]}],
            [{"identity": "Ada", "source_id": "session-1", "embedding": [0, 1]}],
        )

    with pytest.raises(RecognitionEvaluationError, match="exact embedding duplicate"):
        evaluate_embedding_gallery(
            [{"sample_id": "ada", "identity": "Ada", "embedding": [1, 0]}],
            [{"identity": "Ada", "embedding": [1, 0]}],
        )


def test_independent_mode_requires_holdout_provenance_and_accepts_clean_probe():
    result = evaluate_embedding_gallery(
        [{
            "sample_id": "ada-train",
            "identity": "Ada",
            "split": "train",
            "source_id": "enrollment-1",
            "embedding": [1, 0, 0],
        }],
        [{
            "sample_id": "ada-holdout",
            "identity": "Ada",
            "split": "holdout",
            "source_id": "holdout-1",
            "embedding": [0.99, 0.1, 0],
        }],
        threshold=0.8,
        min_margin=0.0,
        require_independent_probes=True,
    )
    assert result["status"] == "MEASURED"
    assert result["accuracy"] == 1.0

    with pytest.raises(RecognitionEvaluationError, match="holdout or adversarial"):
        evaluate_embedding_gallery(
            [{"sample_id": "ada", "identity": "Ada", "embedding": [1, 0, 0]}],
            [{"identity": "Ada", "split": "train", "source_id": "probe", "embedding": [0.99, 0.1, 0]}],
            require_independent_probes=True,
        )
