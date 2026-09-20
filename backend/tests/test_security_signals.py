from core.security import SecuritySignalEngine


def face(state="UNRESOLVED", name="UNKNOWN"):
    return {"oid": 1, "identity_state": state, "name": name}


def test_restricted_zone_entry_is_debounced_and_exits_once():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ENABLED": True,
            "ZONES": [{"id": "lab", "name": "Lab", "bounds": [10, 10, 100, 100], "severity": 2}],
        }
    })
    assert engine.capability_state()["intrusion"] == "AVAILABLE"
    assert engine.update({1: (0, 0)}, faces_info=[face()], now=1.0) == []
    events = engine.update({1: (20, 20)}, faces_info=[face()], now=2.0)
    assert [event[0] for event in events] == ["ZONE_ENTRY", "ZONE_INTRUSION"]
    assert engine.update({1: (30, 30)}, faces_info=[face()], now=3.0) == []
    events = engine.update({1: (0, 0)}, faces_info=[face()], now=4.0)
    assert [event[0] for event in events] == ["ZONE_EXIT"]


def test_loitering_requires_dwell_persistence():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "gate", "bounds": [0, 0, 100, 100], "loitering_seconds": 2}],
        }
    })
    engine.update({1: (50, 50)}, faces_info=[face()], now=1.0)
    assert engine.update({1: (50, 50)}, faces_info=[face()], now=2.9) == []
    events = engine.update({1: (50, 50)}, faces_info=[face()], now=3.1)
    assert any(event[0] == "LOITERING" for event in events)


def test_running_and_evacuation_require_sustained_motion():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": 10, "CONFIRM_SECONDS": 0.2},
            "EVACUATION": {"MIN_PEOPLE": 2, "AVERAGE_SPEED_THRESHOLD_PX_SEC": 5,
                           "CONFIRM_SECONDS": 0.2, "COOLDOWN_SECONDS": 5},
        }
    })
    engine.update({1: (0, 0), 2: (0, 100)}, now=1.0)
    engine.update({1: (4, 0), 2: (4, 100)}, now=1.1)
    events = engine.update({1: (8, 0), 2: (8, 100)}, now=1.4)
    types = [event[0] for event in events]
    assert "RUNNING" in types
    assert "EVACUATION_ALERT" in types


def test_running_can_use_frame_normalized_speed_and_reports_mode():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": 100, "CONFIRM_SECONDS": 0.1},
        }
    })
    engine.update({1: (0, 0)}, now=1.0, frame_size=(100, 100))
    engine.update({1: (12, 0)}, now=1.1, frame_size=(100, 100))
    events = engine.update({1: (24, 0)}, now=1.3, frame_size=(100, 100))
    running = next(event for event in events if event[0] == "RUNNING")
    assert running[4]["threshold_mode"] == "frame_normalized"
    assert running[4]["speed_normalized"] is not None
    assert running[4]["frame_diagonal_px"] == 141.42


def test_evacuation_can_use_frame_normalized_speed_and_reports_mode():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": 100000},
            "EVACUATION": {
                "MIN_PEOPLE": 2,
                "AVERAGE_SPEED_THRESHOLD_PX_SEC": 10,
                "CONFIRM_SECONDS": 0.1,
                "COOLDOWN_SECONDS": 5,
            },
        }
    })
    engine.update({1: (0, 0), 2: (0, 100)}, now=1.0, frame_size=(100, 100))
    engine.update({1: (4, 0), 2: (4, 100)}, now=1.1, frame_size=(100, 100))
    events = engine.update(
        {1: (8, 0), 2: (8, 100)}, now=1.4, frame_size=(100, 100))

    evacuation = next(event for event in events if event[0] == "EVACUATION_ALERT")
    assert evacuation[4]["threshold_mode"] == "frame_normalized"
    assert evacuation[4]["average_speed_normalized"] is not None
    assert evacuation[4]["average_speed_px_sec"] > 0
    assert evacuation[4]["frame_diagonal_px"] == 141.42


def test_ppe_is_unavailable_without_a_compatible_model():
    engine = SecuritySignalEngine({"SECURITY": {"PPE": {"ENABLED": True, "AVAILABLE": False}}})
    assert engine.capability_state()["ppe"] == "NOT_CONFIGURED"
    assert engine.update({1: (50, 50)}, object_detections=[{
        "class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.99,
    }], now=1.0) == []


def test_ppe_persistence_does_not_cross_tracker_generations():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "PPE": {
                "ENABLED": True,
                "AVAILABLE": True,
                "REQUIRED_OBJECTS": ["helmet"],
                "PERSISTENCE_SECONDS": 1.0,
            }
        }
    })
    person = {"class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.99}

    engine.update(
        {1: (50, 50)},
        faces_info=[{"oid": 1, "track_generation": 1, "identity_state": "UNRESOLVED"}],
        object_detections=[person], now=1.0, track_generations={1: 1})
    events = engine.update(
        {1: (50, 50)},
        faces_info=[{"oid": 1, "track_generation": 1, "identity_state": "UNRESOLVED"}],
        object_detections=[person], now=2.1, track_generations={1: 1})
    assert any(event[0] == "PPE_VIOLATION" for event in events)

    engine.update(
        {1: (50, 50)},
        faces_info=[{"oid": 1, "track_generation": 2, "identity_state": "UNRESOLVED"}],
        object_detections=[person], now=2.2, track_generations={1: 2})
    fresh_generation = engine.update(
        {1: (50, 50)},
        faces_info=[{"oid": 1, "track_generation": 2, "identity_state": "UNRESOLVED"}],
        object_detections=[person], now=3.0, track_generations={1: 2})
    assert not any(event[0] == "PPE_VIOLATION" for event in fresh_generation)


def test_zone_policy_can_allow_a_role_and_respect_an_active_schedule():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{
                "id": "staff", "bounds": [0, 0, 100, 100],
                "allowed_roles": ["staff"],
                "schedule": {"weekdays": [0], "start": "09:00", "end": "17:00"},
            }],
        }
    })
    allowed = face(state="CONFIRMED", name="Ada")
    engine.set_roster_roles({"Ada": "Staff"})
    engine.update({1: (150, 150)}, faces_info=[allowed], now=1.0,
                  wallclock="2026-09-07T10:00:00")
    events = engine.update({1: (10, 10)}, faces_info=[allowed], now=2.0,
                           wallclock="2026-09-07T10:00:00")
    assert [event[0] for event in events] == ["ZONE_ENTRY"]

    scheduled = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{
                "id": "staff", "bounds": [0, 0, 100, 100],
                "schedule": {"weekdays": [0], "start": "09:00", "end": "17:00"},
            }],
        }
    })
    assert scheduled.update({1: (150, 150)}, faces_info=[face()], now=1.0,
                             wallclock="2026-09-07T18:00:00") == []
    scheduled.update({1: (150, 150)}, faces_info=[face()], now=2.0,
                     wallclock="2026-09-07T10:00:00")
    events = scheduled.update({1: (20, 20)}, faces_info=[face()], now=3.0,
                              wallclock="2026-09-07T10:00:00")
    assert [event[0] for event in events] == ["ZONE_ENTRY", "ZONE_INTRUSION"]


def test_malformed_zone_configuration_fails_closed_and_is_reported():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ENABLED": True,
            "ZONES": [{
                "id": "restricted",
                "bounds": [0, 0, "not-a-number", 100],
                "schedule": {"weekdays": [9]},
            }],
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": "nan"},
        }
    })
    capabilities = engine.capability_state()
    assert capabilities["configuration"] == "INVALID"
    assert capabilities["intrusion"] == "NOT_CONFIGURED"
    assert engine.update({1: (20, 20)}, faces_info=[], now=1.0) == []


def test_invalid_security_policy_disables_operational_capabilities():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "lab", "bounds": [0, 0, 100, 100]}],
            "RUNNING": {"SPEED_THRESHOLD_PX_SEC": "bad"},
        }
    })

    capabilities = engine.capability_state()
    assert capabilities["configuration"] == "INVALID"
    assert capabilities["policy_valid"] is False
    assert capabilities["intrusion"] == "NOT_CONFIGURED"
    assert capabilities["running"] == "DISABLED"
    assert engine.update({1: (0, 0), 2: (20, 20)}, now=1.0) == []


def test_invalid_zone_schedule_is_not_treated_as_active():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{
                "id": "restricted",
                "bounds": [0, 0, 100, 100],
                "schedule": {"start": "not-a-clock"},
            }],
        }
    })

    assert engine.capability_state()["configuration"] == "INVALID"
    assert engine.capability_state()["intrusion"] == "NOT_CONFIGURED"
    assert engine.zones == []


def test_malformed_ppe_rows_do_not_interrupt_security_updates():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "PPE": {
                "ENABLED": True,
                "AVAILABLE": True,
                "REQUIRED_OBJECTS": ["helmet"],
            }
        }
    })

    assert engine.update(
        {1: (50, 50)},
        object_detections=[
            {"class_name": "person", "bbox": [0, 0, "bad", 100]},
            {"class_name": "helmet", "bbox": [0, 0, 100, 100], "confidence": "bad"},
        ],
        now=1.0,
    ) == []


def test_zone_and_dwell_state_are_scoped_to_the_camera():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "lab", "camera_id": "cam_a", "bounds": [0, 0, 100, 100]},
                      {"id": "office", "camera_id": "cam_b", "bounds": [0, 0, 100, 100]}],
        }
    })

    assert engine.update(
        {1: (150, 150)}, faces_info=[face()], now=1.0, camera_id="cam_a"
    ) == []
    assert engine.update(
        {1: (150, 150)}, faces_info=[face()], now=1.0, camera_id="cam_b"
    ) == []
    assert [event[0] for event in engine.update(
        {1: (20, 20)}, faces_info=[face()], now=2.0, camera_id="cam_a"
    )] == ["ZONE_ENTRY", "ZONE_INTRUSION"]
    assert [event[0] for event in engine.update(
        {1: (20, 20)}, faces_info=[face()], now=2.0, camera_id="cam_b"
    )] == ["ZONE_ENTRY", "ZONE_INTRUSION"]
    assert engine.update({1: (20, 20)}, faces_info=[face()], now=2.0, camera_id="cam_a") == []
    assert engine.update({1: (20, 20)}, faces_info=[face()], now=2.0, camera_id="cam_b") == []


def test_zone_policy_replacement_is_atomic_and_resets_state():
    engine = SecuritySignalEngine({
        "SECURITY": {
            "ZONES": [{"id": "old", "bounds": [0, 0, 100, 100]},
                      {"id": "second", "bounds": [200, 200, 300, 300]}],
        }
    })
    engine.update({1: (50, 50)}, faces_info=[face()], now=1.0)
    assert engine._tracks

    rejected = engine.replace_zones([{"id": "bad", "bounds": [0, 0, "x", 10]}])
    assert rejected["ok"] is False
    assert [zone.zone_id for zone in engine.zones] == ["old", "second"]
    assert engine._tracks

    accepted = engine.replace_zones([{"id": "new", "bounds": [10, 10, 20, 20]}])
    assert accepted["ok"] is True
    assert accepted["count"] == 1
    assert accepted["zones"][0]["id"] == "new"
    assert engine._tracks == {}
    assert engine._last_event == {}


def test_zone_policy_rejects_duplicate_ids():
    engine = SecuritySignalEngine({"SECURITY": {"ZONES": []}})

    result = engine.replace_zones([
        {"id": "restricted", "bounds": [0, 0, 10, 10]},
        {"id": "restricted", "bounds": [20, 20, 30, 30]},
    ])

    assert result["ok"] is False
    assert "duplicate zone id" in result["issues"][0]
    assert engine.zones == []


def test_zone_policy_rejects_unknown_shape():
    engine = SecuritySignalEngine({"SECURITY": {"ZONES": []}})

    result = engine.replace_zones([{"id": "restricted", "shape": "circle", "bounds": [0, 0, 10, 10]}])

    assert result["ok"] is False
    assert "shape must be rect or polygon" in result["issues"][0]
