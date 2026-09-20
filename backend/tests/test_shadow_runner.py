import pytest

from core.model_adapters import CallableModelAdapter, ModelSpec
from core.shadow_runner import ShadowEvaluationError, run_shadow_evaluation


def _adapter(name, mode, output):
    return CallableModelAdapter(
        ModelSpec(name, "object_detection", capability="AVAILABLE", execution_mode=mode),
        lambda _frame, _context: output,
    )


def test_shadow_runner_compares_normalized_outputs_without_decisions():
    baseline = _adapter("baseline", "active", [{"class_name": "car", "confidence": 0.8}])
    candidate = _adapter("candidate", "shadow", [{"class_name": "car", "confidence": 0.9}])
    report = run_shadow_evaluation([
        {"frame": b"frame", "context": {"source_frame_id": 4}},
    ], baseline, candidate)
    assert report["status"] == "MEASURED"
    assert report["disagreement_count"] == 0
    assert report["decision_effect"] == "NONE"
    assert "attendance" not in report
    assert report["candidate"]["execution_mode"] == "shadow"


def test_shadow_runner_reports_candidate_disagreement():
    baseline = _adapter("baseline", "active", [{"class_name": "car"}])
    candidate = _adapter("candidate", "shadow", [{"class_name": "truck"}])
    report = run_shadow_evaluation([{"frame": 1}], baseline, candidate)
    assert report["disagreement_count"] == 1
    assert report["disagreements"][0]["candidate_only"][0][1] == "truck"


def test_shadow_runner_requires_shadow_candidate_and_valid_input():
    baseline = _adapter("baseline", "active", [])
    active_candidate = _adapter("candidate", "active", [])
    with pytest.raises(ShadowEvaluationError, match="execution_mode"):
        run_shadow_evaluation([], baseline, active_candidate)
    shadow = _adapter("candidate", "shadow", [])
    with pytest.raises(ShadowEvaluationError, match="frame field"):
        run_shadow_evaluation([{"context": {}}], baseline, shadow)
