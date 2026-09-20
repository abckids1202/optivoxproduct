import pytest

from core.model_adapters import CallableModelAdapter, ModelAdapterRegistry, ModelSpec
from core.tracking_adapter import TrackingAdapter


def test_model_registry_exposes_capability_state_and_latency_without_running_models():
    registry = ModelAdapterRegistry(history_size=4)
    registry.register(ModelSpec(
        "plate_detector_ocr",
        "license_plate_reading",
        capability="NOT_CONFIGURED",
        execution_mode="shadow",
        privacy_class="restricted_local",
    ))
    registry.record_inference("plate_detector_ocr", 12.0)
    snapshot = registry.snapshot()
    model = snapshot["models"][0]
    assert model["state"] == "NOT_CONFIGURED"
    assert model["execution_mode"] == "shadow"
    assert model["latency"]["p95_ms"] == 12.0
    assert registry.capability_snapshot() == {"plate_detector_ocr": "NOT_CONFIGURED"}
    assert registry.promotion_allowed("plate_detector_ocr") is False


def test_model_promotion_requires_checksum_evaluation_and_promoted_state():
    registry = ModelAdapterRegistry()
    registry.register(ModelSpec(
        "plate", "license_plate_reading", capability="AVAILABLE",
        checksum_verified=True, promotion_status="shadow",
        evaluation_report="eval-1",
    ))
    assert registry.promotion_allowed("plate") is False


def test_adapter_registry_sync_tightens_matching_manifest_metadata():
    registry = ModelAdapterRegistry()
    registry.register(ModelSpec("plate", "license_plate_reading", capability="AVAILABLE"))
    result = registry.apply_registry_snapshot({
        "models": [{
            "name": "plate", "status": "VALID", "promotion_status": "promoted",
            "evaluation_report": "plate-eval", "rollback_target": "plate-old",
        }],
        "issues": [],
    })
    assert result["matched"] == ["plate"]
    assert registry.promotion_allowed("plate") is True
    synced = registry.snapshot()["models"][0]
    assert synced["promotion_status"] == "promoted"
    assert synced["evaluation_report"] == "plate-eval"
    assert synced["checksum_verified"] is True


def test_registry_run_records_adapter_latency_and_failures():
    registry = ModelAdapterRegistry()
    spec = ModelSpec("runner", "toy_detection", capability="AVAILABLE")
    registry.register(spec)
    adapter = CallableModelAdapter(spec, lambda _frame, _context: [{"class_name": "car"}])
    result = registry.run(adapter, object(), {"source_frame_id": 8})
    assert result["status"] == "OK"
    assert result["registry_elapsed_ms"] >= 0
    model = registry.snapshot()["models"][0]
    assert model["calls"] == 1
    assert model["latency"]["samples"] == 1
    registry.replace(ModelSpec(
        "plate", "license_plate_reading", capability="AVAILABLE",
        checksum_verified=True, promotion_status="promoted",
        evaluation_report="eval-1",
    ))
    assert registry.promotion_allowed("plate") is True
    registry.replace(ModelSpec(
        "plate", "license_plate_reading", capability="AVAILABLE",
        checksum_verified=True, promotion_status="rollback",
        evaluation_report="eval-1",
    ))
    assert registry.promotion_allowed("plate") is False


def test_model_registry_fails_runtime_capability_after_repeated_errors():
    registry = ModelAdapterRegistry()
    registry.register(ModelSpec("danger", "danger_detection", capability="AVAILABLE"))
    for _ in range(3):
        registry.record_inference("danger", 4.0, ok=False, error="model crashed")
    model = registry.snapshot()["models"][0]
    assert model["state"] == "FAILED_RUNTIME"
    assert model["failures"] == 3
    assert model["last_error"] == "model crashed"


def test_callable_adapter_is_observation_only_and_never_decides_attendance():
    adapter = CallableModelAdapter(ModelSpec("plate", "plate_ocr", capability="AVAILABLE"))
    result = adapter.infer(None)
    assert result["status"] == "NOT_CONFIGURED"
    assert result["observations"] == []
    assert "attendance" not in result

    active = CallableModelAdapter(
        ModelSpec("toy", "toy_detection", capability="AVAILABLE"),
        lambda frame, context: [{"class_name": "object", "confidence": 0.9}],
    )
    result = active.infer(object(), {"source_frame_id": 3})
    assert result["status"] == "OK"
    assert result["observations"][0]["class_name"] == "object"
    assert result["observations"][0]["event_type"] == "TOY_DETECTION"
    assert result["observations"][0]["model_name"] == "toy"
    assert result["observations"][0]["source_frame_id"] == 3
    assert result["execution_mode"] == "active"
    assert "attendance" not in result

    forbidden = CallableModelAdapter(
        ModelSpec("unsafe", "toy_detection", capability="AVAILABLE"),
        lambda _frame, _context: [{"event_type": "OBJECT_DETECTED", "attendance": True}],
    )
    assert forbidden.infer(object())["status"] == "INVALID_OUTPUT"

    aggregate = CallableModelAdapter(ModelSpec(
        "age-band", "aggregate_age_band", capability="AVAILABLE",
        privacy_class="sensitive_aggregate",
    ), lambda _frame, _context: [{"age_band": "adult", "person_name": "Ada"}])
    assert aggregate.infer(object())["status"] == "INVALID_OUTPUT"

    shadow = CallableModelAdapter(ModelSpec(
        "shadow", "toy_detection", capability="AVAILABLE",
        execution_mode="shadow", promotion_status="shadow",
        checksum_verified=True, evaluation_report="eval-1",
    ), lambda _frame, _context: [{"class_name": "candidate"}])
    shadow_result = shadow.infer(object())
    assert shadow_result["status"] == "OK"
    assert shadow_result["execution_mode"] == "shadow"

    unapproved_active = CallableModelAdapter(ModelSpec(
        "unapproved", "toy_detection", capability="AVAILABLE",
        execution_mode="active", promotion_status="shadow",
    ), lambda _frame, _context: [{"class_name": "must-not-run"}])
    assert unapproved_active.infer(object())["status"] == "NOT_APPROVED"

    incomplete_promoted = CallableModelAdapter(ModelSpec(
        "incomplete", "toy_detection", capability="AVAILABLE",
        execution_mode="active", promotion_status="promoted",
        checksum_verified=True,
    ), lambda _frame, _context: [{"class_name": "must-not-run"}])
    assert incomplete_promoted.infer(object())["status"] == "NOT_APPROVED"

    rollback = CallableModelAdapter(ModelSpec(
        "rollback", "toy_detection", capability="AVAILABLE",
        promotion_status="rollback", checksum_verified=True,
        evaluation_report="eval-1",
    ), lambda _frame, _context: [{"class_name": "must-not-run"}])
    assert rollback.infer(object())["status"] == "ROLLBACK"

    disabled = CallableModelAdapter(
        ModelSpec("disabled", "toy_detection", capability="DISABLED"),
        lambda frame, context: [{"class_name": "must-not-run"}],
    )
    assert disabled.infer(object())["status"] == "DISABLED"


def test_tracking_adapter_preserves_backend_boundary_and_errors():
    adapter = TrackingAdapter("centroid", lambda detections: {1: (10, 20)})
    assert adapter.update([], source_frame_id=12) == {1: (10, 20)}
    snapshot = adapter.snapshot({1: (10, 20)})
    assert snapshot["backend"] == "centroid"
    assert snapshot["entity_generation_source"] == "CorrelationCore"
    assert snapshot["update_calls"] == 1
    assert snapshot["last_source_frame_id"] == 12
    assert snapshot["id_switches"] == "NOT_MEASURED"

    sequence = iter(({1: (10, 20)}, {2: (11, 20)}))
    continuity = TrackingAdapter("centroid", lambda _detections: next(sequence))
    continuity.update([], source_frame_id=1)
    continuity.update([], source_frame_id=2)
    assert continuity.snapshot()["membership_changes"] == 2

    failing = TrackingAdapter("broken", lambda _: (_ for _ in ()).throw(RuntimeError("tracker down")))
    with pytest.raises(RuntimeError, match="tracker down"):
        failing.update([])
    assert failing.snapshot()["status"] == "AVAILABLE"
    assert failing.snapshot()["failures"] == 1
