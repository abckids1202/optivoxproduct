from core.model_adapters import ModelSpec
from core.vehicle_adapters import PlateOCRAdapter, PlateObservation, PlateTemporalVoter


def test_plate_voter_requires_repeated_consistent_readings():
    voter = PlateTemporalVoter(min_hits=3, min_confidence=0.8)
    assert voter.add("AB 1234", 0.95).status == "PLATE_READ_UNCERTAIN"
    assert voter.add("AB 1235", 0.95).status == "PLATE_READ_UNCERTAIN"
    assert voter.add("AB 1234", 0.9).status == "PLATE_READ_UNCERTAIN"
    result = voter.add("AB 1234", 0.92)
    assert result.status == "PLATE_READ"
    assert result.text_local == "AB 1234"
    assert result.text_hash


def test_plate_observation_keeps_raw_text_local():
    event = PlateObservation(
        "PLATE_READ", track_id=4, text_local="AB 1234", text_hash="hash",
        confidence=0.91, source_frame_id=12,
    ).to_local_event("cam_a")
    assert event["plate_text_local"] == "AB 1234"
    assert event["privacy_class"] == "restricted_local"
    assert event["entity_id"] == "cam_a:vehicle:4"


def test_plate_adapter_fails_closed_without_model():
    adapter = PlateOCRAdapter(ModelSpec(
        "plate", "license_plate_reading", capability="NOT_CONFIGURED"))
    result = adapter.infer(object(), {"track_id": 1, "source_frame_id": 7})
    assert result.status == "PLATE_READ_UNCERTAIN"
    assert result.reason == "not_configured"


def test_plate_adapter_rejects_invalid_or_low_confidence_output():
    adapter = PlateOCRAdapter(
        ModelSpec("plate", "license_plate_reading", capability="AVAILABLE"),
        infer_fn=lambda _crop, _context: {"text": "???", "confidence": 0.99},
    )
    result = adapter.infer(object())
    assert result.status == "PLATE_READ_UNCERTAIN"
    assert result.reason == "invalid_plate_text"


def test_plate_adapter_temporal_state_is_scoped_to_vehicle_track():
    adapter = PlateOCRAdapter(
        ModelSpec("plate", "license_plate_reading", capability="AVAILABLE"),
        infer_fn=lambda _crop, context: {
            "text": "AB 1234", "confidence": 0.95,
            "track_id": context["track_id"],
        },
    )
    assert adapter.infer(object(), {"track_id": 1}).status == "PLATE_READ_UNCERTAIN"
    assert adapter.infer(object(), {"track_id": 1}).status == "PLATE_READ"
    assert adapter.infer(object(), {"track_id": 2}).status == "PLATE_READ_UNCERTAIN"
