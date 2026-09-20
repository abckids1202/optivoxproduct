import json
import json
import threading

import pytest

from core.data_registry import DatasetRecord, DatasetRegistry
from core.correlation import CorrelationCore
from core.evaluation import binary_metrics, event_metrics, mean_absolute_error, tracking_id_switches
from core.vehicle_intelligence import VehicleIntelligence
from core.notification_policy import notification_decision
from core.event_envelope import EventEnvelope
from core.edge_sync import EdgeEventOutbox, OutboxSecurityError, verify_sync_batch
from core.demographics import age_band


def test_age_estimation_is_coarse_and_rejects_invalid_values():
    assert age_band(10) == "child"
    assert age_band(16.5) == "teen"
    assert age_band(38) == "adult"
    assert age_band("not-an-age") is None
    assert age_band(140) is None


def test_crowd_snapshot_labels_detector_density_as_experimental():
    from main import CrowdIntelligence

    crowd = CrowdIntelligence({"CROWD_INTELLIGENCE": {
        "ENABLED": True, "HEATMAP_GRID": (4, 4),
    }})
    crowd.update({1: (10, 10), 2: (10, 10), 3: (90, 90)}, frame_size=(100, 100))
    snapshot = crowd.snapshot()

    assert snapshot["status"] == "EXPERIMENTAL"
    assert snapshot["method"] == "tracked_person_occupancy"
    assert snapshot["people_count"] == 3
    assert snapshot["perspective_correction"] == "NOT_CONFIGURED"
    assert max(cell["people"] for cell in snapshot["density_cells"]) == 2


def test_dataset_registry_rejects_split_leakage_and_escape(tmp_path):
    registry = DatasetRegistry(tmp_path, [
        DatasetRecord("live-1", "faces/a.jpg", "face", "train", label="Ada", source_id="ada-1", provenance="consented"),
        DatasetRecord("live-1-holdout", "faces/b.jpg", "face", "holdout", label="Ada", source_id="ada-1", provenance="consented"),
        DatasetRecord("escape", "../secret.jpg", "face", "train", label="Ada", provenance="consented"),
    ])
    issues = registry.validate()
    assert any("split leakage" in issue for issue in issues)
    assert any("escapes dataset root" in issue for issue in issues)


def test_dataset_registry_rejects_broken_augmentation_lineage(tmp_path):
    registry = DatasetRegistry(tmp_path, [
        DatasetRecord("source", "faces/a.jpg", "face", "train", label="Ada", provenance="consented"),
        DatasetRecord("derived", "faces/b.jpg", "face", "holdout", label="Ada",
                      derived_from="source", transform="brightness_contrast", provenance="consented"),
        DatasetRecord("orphan", "faces/c.jpg", "face", "train", label="Ada",
                      derived_from="missing", transform="crop_in", provenance="consented"),
    ])
    issues = registry.validate()
    assert any("derived record split" in issue for issue in issues)
    assert any("derived_from does not reference" in issue for issue in issues)


def test_dataset_registry_round_trip_preserves_provenance(tmp_path):
    path = tmp_path / "dataset.json"
    registry = DatasetRegistry(tmp_path, [DatasetRecord(
        "frame-1", "frames/one.jpg", "intrusion", "adversarial",
        label="zone_entry", provenance="licensed", metadata={"camera": "gate"},
    )])
    registry.save(path)
    loaded = DatasetRegistry.load(path)
    assert loaded.records[0].provenance == "licensed"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_model_registry_requires_metadata_checksum_and_promotion_evidence(tmp_path):
    import hashlib
    from core.model_registry import ModelRegistry

    model = tmp_path / "models" / "face.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model-artifact")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = tmp_path / "models" / "model_registry.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": [{
            "name": "face-recognizer",
            "task": "face_embedding",
            "version": "1.0.0",
            "path": "models/face.onnx",
            "sha256": digest,
            "source": "local-validation",
            "license": "internal",
            "evaluation_report": "reports/face-1.json",
            "promotion_status": "shadow",
        }],
    }), encoding="utf-8")
    result = ModelRegistry(tmp_path, manifest).load().snapshot()
    assert result["status"] == "VALID"
    assert result["models"][0]["status"] == "VALID"
    registry = ModelRegistry(tmp_path, manifest).load()
    assert registry.promotion_allowed("face-recognizer", "1.0.0") is False
    assert registry.promotion_allowed("face-recognizer", "1.0.0", require_promoted=False) is True

    model.write_bytes(b"tampered")
    result = ModelRegistry(tmp_path, manifest).load().snapshot()
    assert result["status"] == "INVALID"
    assert any("checksum mismatch" in issue for issue in result["issues"])


def test_model_registry_rejects_absolute_or_traversing_paths(tmp_path):
    from core.model_registry import ModelRegistry

    manifest = tmp_path / "model_registry.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": [{
            "name": "unsafe", "task": "test", "version": "1",
            "path": "../outside.onnx", "sha256": "0" * 64,
            "source": "test", "license": "test", "evaluation_report": "test",
        }],
    }), encoding="utf-8")
    result = ModelRegistry(tmp_path, manifest).load().snapshot()
    assert result["status"] == "INVALID"
    assert any("escapes registry root" in issue for issue in result["issues"])


def test_evaluation_reports_false_accepts_rejects_and_tracking_switches():
    result = binary_metrics([True, True, False, False], [0.9, 0.2, 0.8, 0.1], 0.5).to_dict()
    assert result["false_accept_rate"] == 0.5
    assert result["false_reject_rate"] == 0.5
    assert event_metrics(["a", "b"], ["a", "c"], duration_hours=2)["false_alerts_per_camera_hour"] == 0.5
    assert mean_absolute_error([10, 20], [12, 17]) == 2.5
    assert tracking_id_switches([(1, "person-1", 4), (2, "person-1", 4), (3, "person-1", 8)])["id_switches"] == 1


def test_vehicle_speed_is_unavailable_without_calibration():
    intelligence = VehicleIntelligence({"SPEED_LIMIT_KMH": 1, "SPEED_CONFIRM_SECONDS": 0.1})
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    intelligence.update([detection], now=0.0, source_frame_id=1)
    events = intelligence.update([{**detection, "bbox": [100, 0, 110, 10]}], now=1.0, source_frame_id=2)
    assert intelligence.capability_state()["vehicle_speed"] == "NOT_CONFIGURED"
    assert not any(event[0] == "SPEED_THRESHOLD_EXCEEDED" for event in events)


def test_vehicle_calibration_accepts_only_valid_finite_homography():
    intelligence = VehicleIntelligence({})
    assert intelligence.set_calibration({"homography": [[1, 0], [0, 1]], "image_width": 640, "image_height": 480}) is False
    assert intelligence.set_calibration({
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "image_width": 640,
        "image_height": 480,
    }) is True
    assert intelligence.capability_state()["vehicle_speed"] == "AVAILABLE"


def test_calibrated_vehicle_speed_and_plate_voting_are_conservative():
    intelligence = VehicleIntelligence({
        "SPEED_LIMIT_KMH": 20,
        "SPEED_CONFIRM_SECONDS": 0.2,
        "CALIBRATION": {
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "image_width": 640,
            "image_height": 480,
        },
    })
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    intelligence.update([detection], now=0.0, source_frame_id=1, frame_size=(320, 240))
    intelligence.update([{**detection, "bbox": [10, 0, 20, 10]}], now=1.0, source_frame_id=2, frame_size=(320, 240))
    events = intelligence.update([{**detection, "bbox": [20, 0, 30, 10]}], now=1.3, source_frame_id=3, frame_size=(320, 240))
    assert intelligence.capability_state()["vehicle_speed"] == "AVAILABLE"
    assert any(event[0] == "SPEED_THRESHOLD_EXCEEDED" for event in events)
    assert any(event[0] == "SPEED_ESTIMATE" and event[4]["calibrated"] for event in events)
    assert intelligence.ingest_plate(1, "ab 123 cd", 0.9, 1.0, 2) is None
    assert intelligence.ingest_plate(1, "ab 123 cd", 0.9, 1.1, 3) is None
    plate_event = intelligence.ingest_plate(1, "ab 123 cd", 0.9, 1.2, 4)
    assert plate_event[0] == "PLATE_READ"
    assert plate_event[4]["privacy_class"] == "restricted_local"
    assert plate_event[4]["entity_id"] == "cam_0:vehicle:1"


def test_vehicle_calibration_maps_resized_detector_coordinates():
    intelligence = VehicleIntelligence({
        "CALIBRATION": {
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "image_width": 640,
            "image_height": 480,
        },
    })
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    intelligence.update([detection], now=0.0, source_frame_id=1, frame_size=(320, 240))
    assert intelligence.tracks[1].ground_point == (10.0, 20.0)


def test_vehicle_timeout_emits_one_attributable_exit_event():
    intelligence = VehicleIntelligence({"TRACK_TTL_SEC": 1.0})
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    entered = intelligence.update([detection], now=0.0, source_frame_id=10)
    assert [event[0] for event in entered] == ["VEHICLE_ENTERED"]
    exited = intelligence.update([], now=1.1, source_frame_id=11)
    assert [event[0] for event in exited] == ["VEHICLE_EXITED"]
    assert exited[0][1] == "VEHICLE_1"
    assert exited[0][4]["exit_reason"] == "track_timeout"
    assert exited[0][4]["entity_id"] == "cam_0:vehicle:1"
    assert intelligence.update([], now=2.2, source_frame_id=12) == []


def test_vehicle_velocity_prediction_preserves_fast_track_identity():
    intelligence = VehicleIntelligence({
        "MAX_MATCH_DISTANCE_PX": 45,
        "MAX_PREDICTION_SEC": 1.0,
    })
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9,
                 "bbox": [0, 0, 10, 10]}
    intelligence.update([detection], now=0.0, source_frame_id=1)
    intelligence.update([{**detection, "bbox": [20, 0, 30, 10]}], now=1.0, source_frame_id=2)
    events = intelligence.update([{**detection, "bbox": [70, 0, 80, 10]}], now=2.0, source_frame_id=3)

    assert set(intelligence.tracks) == {1}
    assert not any(event[0] == "VEHICLE_ENTERED" for event in events)
    assert intelligence.tracks[1].velocity[0] > 0


def test_unreadable_plate_is_explicit_uncertainty_without_text():
    intelligence = VehicleIntelligence({})
    detection = {"class_name": "car", "category": "vehicle", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
    intelligence.update([detection], now=0.0, source_frame_id=10)
    event = intelligence.plate_unreadable(1, "blurred_crop", 11)
    assert event[0] == "PLATE_READ_UNCERTAIN"
    assert "plate_text" not in str(event[4]).lower()
    assert event[4]["reason"] == "blurred_crop"


def test_plate_text_is_removed_before_operational_persistence():
    from main import _sanitize_event_metadata

    safe = _sanitize_event_metadata({
        "plate_text_local": "AB 123 CD",
        "plate_hash": "hash",
        "ocr_confidence": 0.91,
    })
    assert "plate_text_local" not in safe
    assert safe["plate_hash"] == "hash"
    assert safe["plate_data_redacted"] is True


def test_runtime_state_reports_actual_camera_health(tmp_path):
    import json
    from types import SimpleNamespace
    from main import _publish_runtime_state

    vision = SimpleNamespace(
        _last_faces_seen=[], _last_objects_seen=[], _last_vehicle_state={},
        _last_crowd_state={},
        _last_demographics_state={"status": "EXPERIMENTAL", "age_bands": {"adult": 1}, "samples": 1},
        _last_correlation_state={}, sync_outbox=None,
        security_signals=None,
    )
    _publish_runtime_state(
        str(tmp_path), vision, None, "cam-1", "Test room", 0.0,
        0.0, "2026-09-17T10:00:00+00:00", [],
        camera_status="disconnected",
    )
    state = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    assert state["camera"]["status"] == "disconnected"
    assert state["demographics"]["decision_use"] == "aggregate_only"
    assert state["demographics"]["age_bands"] == {"adult": 1}


def test_vehicle_track_namespace_cannot_inherit_person_identity():
    core = CorrelationCore(identity_confirmation_observations=1)
    core.update(
        tracked={1: (50, 50)},
        faces_info=[{
            "oid": 1, "bbox": (0, 0, 100, 100), "name": "Ada",
            "identity_state": "CONFIRMED", "confidence": 0.95,
            "quality_score": 95, "quality_ok": True, "liveness_status": "REAL",
        }],
    )
    result = core.correlate_security_events([
        ("VEHICLE_ENTERED", "VEHICLE_1", 0.9, "Vehicle entered", {
            "track_id": 1, "entity_id": "cam_0:vehicle:1", "entity_type": "vehicle",
        }),
    ])
    assert result["security_event_tuples"][0][1] == "VEHICLE_1"
    assert result["security_events"][0]["metadata"]["entity_type"] == "vehicle"
    assert result["security_events"][0]["metadata"]["entity_id"] == "cam_0:vehicle:1"
    assert result["security_events"][0]["metadata"]["identity_state"] == "NOT_APPLICABLE"


def test_external_notification_policy_is_allowlisted_and_sensitive_safe():
    assert notification_decision("ZONE_INTRUSION")["decision"] == "AUTO_SEND"
    assert notification_decision("IDENTITY_CLAIM")["decision"] == "REVIEW_ONLY"
    assert notification_decision("ZONE_INTRUSION", "disciplinary accusation")["decision"] == "APPROVAL_REQUIRED"


def _identity_face(name="Ada", state="CANDIDATE", liveness="REAL"):
    return {
        "oid": 1,
        "bbox": (0, 0, 80, 80),
        "name": name,
        "identity_state": state,
        "confidence": 0.92,
        "current_observation_similarity": 0.92,
        "quality_score": 95,
        "quality_ok": True,
        "liveness_status": liveness,
    }


def test_identity_confirmation_uses_bounded_votes_and_rejects_unknown_contradiction():
    core = CorrelationCore(identity_confirmation_observations=3, identity_confirmation_window=5)
    import time

    now = time.monotonic()
    sequence = ["Ada", "Bea", "Ada", "Ada"]
    states = []
    for index, name in enumerate(sequence):
        state = core.update(
            tracked={1: (40, 40)},
            faces_info=[_identity_face(name)],
            source_frame_id=index + 1,
            observed_at_monotonic=now + index * 0.1,
        )
        states.append(state["entities"][0]["identity"]["state"])
    assert states[-1] == "CONFIRMED"
    assert core.entities.get(1).identity.confirmation_hits == 3

    state = core.update(
        tracked={1: (40, 40)},
        faces_info=[_identity_face("UNKNOWN", "UNRESOLVED")],
        source_frame_id=5,
        observed_at_monotonic=now + 0.5,
    )
    identity = state["entities"][0]["identity"]
    assert identity["state"] == "CONTRADICTED"
    assert state["entities"][0]["attendance_eligibility"] is False


def test_entity_lifecycle_transitions_are_emitted_for_persistence():
    core = CorrelationCore(close_after_sec=2.0)
    import time

    now = time.monotonic()
    first = core.update(tracked={1: (10, 10)}, source_frame_id=1, observed_at_monotonic=now)
    assert first["presence_transitions"][0]["type"] == "opened"

    occluded = core.update(tracked={}, source_frame_id=2, observed_at_monotonic=now + 0.2)
    assert occluded["presence_transitions"][0]["type"] == "occluded"

    closed = core.update(tracked={}, source_frame_id=3, observed_at_monotonic=now + 2.3)
    assert any(item["type"] == "closed" for item in closed["presence_transitions"])


def test_security_generation_change_resets_zone_history():
    from core.security import SecuritySignalEngine

    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "restricted", "bounds": [0, 0, 100, 100]}],
        }
    })
    first = engine.update(
        {1: (150, 150)},
        faces_info=[{**_identity_face(), "oid": 1, "track_generation": 1}],
        now=1.0,
        track_generations={1: 1},
    )
    assert not first
    first_entry = engine.update(
        {1: (20, 20)},
        faces_info=[{**_identity_face(), "oid": 1, "track_generation": 1}],
        now=1.5,
        track_generations={1: 1},
    )
    assert [item[0] for item in first_entry] == ["ZONE_ENTRY", "ZONE_INTRUSION"]
    engine.update(
        {1: (150, 150)},
        faces_info=[{**_identity_face(), "oid": 1, "track_generation": 2}],
        now=2.0,
        track_generations={1: 2},
    )
    second = engine.update(
        {1: (20, 20)},
        faces_info=[{**_identity_face(), "oid": 1, "track_generation": 2}],
        now=2.5,
        track_generations={1: 2},
    )
    assert [item[0] for item in second] == ["ZONE_ENTRY", "ZONE_INTRUSION"]


def test_event_envelope_is_stable_and_privacy_aware():
    event = ("ZONE_INTRUSION", "Ada", 0.95, "Restricted entry", {"zone_id": "lab"})
    first = EventEnvelope.from_security_tuple(
        event,
        camera_id="cam_0",
        occurred_at="2026-09-17T10:00:00+00:00",
        source_frame_id=44,
        entity_id="cam_0:entity:1",
        metadata={"policy_version": "zone-2", "site_id": "site-a", "organization_id": "org-a", "device_id": "edge-a"},
    )
    second = EventEnvelope.from_security_tuple(
        event,
        camera_id="cam_0",
        occurred_at="2026-09-17T10:00:00+00:00",
        source_frame_id=44,
        entity_id="cam_0:entity:1",
        metadata={"policy_version": "zone-2", "site_id": "site-a", "organization_id": "org-a", "device_id": "edge-a"},
    )
    assert first.event_id == second.event_id
    assert first.correlation_id == second.correlation_id
    assert first.to_dict()["privacy_class"] == "operational"
    assert first.to_dict()["site_id"] == "site-a"
    assert first.to_dict()["organization_id"] == "org-a"
    assert first.to_dict()["device_id"] == "edge-a"


def test_event_envelope_keeps_security_zones_in_separate_contexts():
    base = {
        "camera_id": "cam_0",
        "occurred_at": "2026-09-17T10:00:00+00:00",
        "entity_id": "cam_0:entity:1",
        "presence_session_id": "presence-1",
        "metadata": {"policy_version": "zone-2"},
    }
    left = EventEnvelope.from_security_tuple(
        ("ZONE_INTRUSION", "Ada", 0.95, "Restricted entry", {"zone_id": "lab"}),
        **base,
    )
    right = EventEnvelope.from_security_tuple(
        ("ZONE_INTRUSION", "Ada", 0.95, "Restricted entry", {"zone_id": "office"}),
        **base,
    )
    assert left.correlation_id != right.correlation_id


def test_correlation_does_not_fabricate_persisted_presence_session_id():
    core = CorrelationCore(identity_confirmation_observations=1)
    result = core.update(
        tracked={1: (50, 50)},
        faces_info=[{
            "oid": 1,
            "bbox": [40, 40, 60, 60],
            "name": "Ada",
            "identity_state": "CONFIRMED",
            "confidence": 0.96,
            "quality_score": 92,
            "quality_ok": True,
            "liveness_status": "REAL",
        }],
        source_frame_id=12,
        camera_id="cam_0",
        observed_at_monotonic=1.0,
        observed_at_wallclock="2026-09-17T10:00:00+00:00",
        security_events=[(
            "ZONE_INTRUSION", "PERSON", 0.9, "Restricted entry",
            {"track_id": 1, "zone_id": "lab"},
        )],
    )

    event = result["security_events"][0]
    assert event["metadata"]["entity_id"] == "cam_0:entity:1"
    assert event["metadata"]["presence_session_id"] is None
    assert event["metadata"]["presence_session_key"].endswith(":generation:1")
    assert event["envelope"]["presence_session_id"] is None


def test_global_object_event_is_not_attributed_to_only_visible_person(tmp_path):
    from main import _process_runtime_side_effects

    class FakeDatabase:
        def __init__(self):
            self.logged = []

        def upsert_presence_session(self, **kwargs):
            return 44

        def close_stale_presence_sessions(self, _timeout):
            return []

        def get_person_id(self, _name):
            return None

        def get_active_person_id(self, _name):
            return None

        def log_event(self, **kwargs):
            self.logged.append(kwargs)
            return 91

    class FakeAttendance:
        def reconcile(self, force=False):
            return []

    class FakeAlerts:
        def check_and_alert(self, **_kwargs):
            return None

    db = FakeDatabase()
    _process_runtime_side_effects(
        {
            "frame_id": 12,
            "completed_at": "2026-09-17T10:00:00+00:00",
            "faces_info": [],
            "events": [(
                "DANGEROUS_OBJECT", "knife", 0.9, "Object detected",
                {"entity_type": "object", "correlation_id": "corr-object"},
            )],
            "correlation": {
                "enabled": True,
                "decision_boundary": "CORRELATION_CORE",
                "entities": [{
                    "entity_id": "cam_0:entity:1", "track_id": 1,
                    "track_generation": 1,
                    "identity": {"state": "CONFIRMED", "confirmed_name": "Ada", "best_score": 0.9},
                    "liveness": {"state": "REAL"},
                    "lifecycle_state": "ACTIVE",
                    "attendance_eligibility": True,
                }],
            },
        },
        db,
        FakeAttendance(),
        FakeAlerts(),
        tmp_path,
        "cam_0",
        "test",
        threading.Lock(),
        lambda: True,
    )

    assert db.logged[0]["entity_id"] is None
    assert db.logged[0]["presence_session_id"] is None


def test_raw_person_event_cannot_attach_inactive_or_unconfirmed_identity(tmp_path):
    from main import _process_runtime_side_effects

    class FakeDatabase:
        def __init__(self):
            self.logged = []

        def upsert_presence_session(self, **kwargs):
            return 44

        def close_stale_presence_sessions(self, _timeout):
            return []

        def get_active_person_id(self, _name):
            # The test proves the event path must not ask SQLite to authorize
            # the display target when correlation has already rejected it.
            return 7

        def log_event(self, **kwargs):
            self.logged.append(kwargs)
            return 91

    class FakeAttendance:
        def reconcile(self, force=False):
            return []

    class FakeAlerts:
        def check_and_alert(self, **_kwargs):
            return None

    db = FakeDatabase()
    _process_runtime_side_effects(
        {
            "frame_id": 12,
            "completed_at": "2026-09-17T10:00:00+00:00",
            "faces_info": [],
            "events": [(
                "ZONE_INTRUSION", "Ada", 0.9, "Restricted entry",
                {"track_id": 1, "entity_type": "person"},
            )],
            "correlation": {
                "enabled": True,
                "decision_boundary": "CORRELATION_CORE",
                "entities": [{
                    "entity_id": "cam_0:entity:1", "track_id": 1,
                    "track_generation": 1,
                    "identity": {
                        "state": "CANDIDATE", "confirmed_name": "Ada",
                        "best_score": 0.9,
                    },
                    "roster_match": False,
                    "liveness": {"state": "REAL"},
                    "lifecycle_state": "ACTIVE",
                    "attendance_eligibility": False,
                }],
            },
        },
        db,
        FakeAttendance(),
        FakeAlerts(),
        tmp_path,
        "cam_0",
        "test",
        threading.Lock(),
        lambda: True,
    )

    assert db.logged[0]["person_id"] is None
    assert db.logged[0]["entity_id"] == "cam_0:entity:1"


def test_edge_outbox_is_idempotent_hash_chained_and_minimized(tmp_path):
    outbox = EdgeEventOutbox(tmp_path / "outbox.ndjson", "edge-test", "sync-secret")
    event = {"event_id": "evt-1", "event_type": "ZONE_INTRUSION", "severity": 2}
    first = outbox.enqueue(event)
    second = outbox.enqueue(event)
    assert first["record_hash"] == second["record_hash"]
    assert outbox.verify_chain()["valid"] is True
    assert len(outbox.pending()) == 1
    assert outbox.acknowledge(["evt-1"]) == 1
    assert outbox.pending() == []


def test_edge_outbox_rejects_biometric_and_local_plate_data(tmp_path):
    outbox = EdgeEventOutbox(tmp_path / "outbox.ndjson", "edge-test")
    with pytest.raises(OutboxSecurityError):
        outbox.enqueue({"event_type": "identity", "embedding": [0.1, 0.2]})
    with pytest.raises(OutboxSecurityError):
        outbox.enqueue({"event_type": "plate", "plate_text_local": "AB123"})


def test_edge_outbox_fails_closed_on_malformed_queue_line(tmp_path):
    path = tmp_path / "outbox.ndjson"
    outbox = EdgeEventOutbox(path, "edge-test")
    outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_DISCONNECTED"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{malformed-json}\n")
    result = outbox.verify_chain()
    assert result["valid"] is False
    assert any("invalid JSON" in issue for issue in result["issues"])


def test_edge_sync_batch_requires_contiguous_provenance_and_rejects_replay(tmp_path):
    outbox = EdgeEventOutbox(
        tmp_path / "outbox.ndjson", "edge-test", "sync-secret",
        site_id="site-a", organization_id="org-a",
    )
    first = outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})
    second = outbox.enqueue({"event_id": "evt-2", "event_type": "ZONE_ENTRY"})

    valid = verify_sync_batch(
        [first, second], expected_device_id="edge-test", signing_secret="sync-secret",
        expected_site_id="site-a", expected_organization_id="org-a",
    )
    assert valid["valid"] is True
    assert valid["last_sequence"] == 2

    replay = verify_sync_batch(
        [first], expected_device_id="edge-test", signing_secret="sync-secret",
        last_accepted_sequence=1, expected_previous_hash=first["record_hash"],
        expected_site_id="site-a", expected_organization_id="org-a",
    )
    assert replay["valid"] is False
    assert any("replay" in issue for issue in replay["issues"])


def test_edge_sync_batch_rejects_device_scope_and_sequence_gap(tmp_path):
    outbox = EdgeEventOutbox(
        tmp_path / "outbox.ndjson", "edge-test", "sync-secret",
        site_id="site-a", organization_id="org-a",
    )
    first = outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})
    forged = dict(first)
    forged["device_id"] = "other-edge"
    forged["site_id"] = "site-b"
    result = verify_sync_batch(
        [forged], expected_device_id="edge-test", signing_secret="sync-secret",
        expected_site_id="site-a", expected_organization_id="org-a",
    )
    assert result["valid"] is False
    assert any("device mismatch" in issue for issue in result["issues"])
    assert any("site mismatch" in issue for issue in result["issues"])


def test_edge_outbox_checkpoint_advances_only_contiguously(tmp_path):
    outbox = EdgeEventOutbox(tmp_path / "outbox.ndjson", "edge-test", "sync-secret")
    first = outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})
    second = outbox.enqueue({"event_id": "evt-2", "event_type": "CAMERA_HEALTH"})

    assert outbox.acknowledge([second["event_id"]]) == 1
    assert outbox.checkpoint()["sequence"] == 0
    assert outbox.verify_chain()["valid"] is True

    assert outbox.acknowledge([first["event_id"]]) == 1
    checkpoint = outbox.checkpoint()
    assert checkpoint["sequence"] == 2
    assert checkpoint["record_hash"] == second["record_hash"]
    assert outbox.verify_chain()["valid"] is True


def test_edge_outbox_tampered_checkpoint_fails_closed(tmp_path):
    path = tmp_path / "outbox.ndjson"
    outbox = EdgeEventOutbox(path, "edge-test", "sync-secret")
    outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})
    meta_path = path.with_suffix(path.suffix + ".meta")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["next_sequence"] = "not-a-sequence"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = outbox.verify_chain()
    assert result["valid"] is False
    assert any("metadata next sequence" in issue for issue in result["issues"])
