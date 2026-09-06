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


def test_ppe_is_unavailable_without_a_compatible_model():
    engine = SecuritySignalEngine({"SECURITY": {"PPE": {"ENABLED": True, "AVAILABLE": False}}})
    assert engine.capability_state()["ppe"] == "NOT_CONFIGURED"
    assert engine.update({1: (50, 50)}, object_detections=[{
        "class_name": "person", "bbox": [0, 0, 100, 100], "confidence": 0.99,
    }], now=1.0) == []


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
    events = engine.update({1: (10, 10)}, faces_info=[allowed], now=1.0,
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
    assert scheduled.update({1: (10, 10)}, faces_info=[face()], now=1.0,
                             wallclock="2026-09-07T18:00:00") == []
    events = scheduled.update({1: (20, 20)}, faces_info=[face()], now=2.0,
                              wallclock="2026-09-07T10:00:00")
    assert [event[0] for event in events] == ["ZONE_ENTRY", "ZONE_INTRUSION"]
