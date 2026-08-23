import json

from backend.services import runtime_service


def test_capability_handshake_is_exposed(tmp_path, monkeypatch):
    capability_path = tmp_path / "capability.json"
    capability_path.write_text(
        json.dumps({
            "schema_version": 1,
            "runtime_id": "edge-test-01",
            "runtime_version": "2.1.0",
            "pid": 123,
            "started_at": "2026-08-22T10:00:00+00:00",
            "generated_at": "2026-08-22T10:00:01+00:00",
            "capabilities": {"runtime_bridge": True, "web_enrollment": True},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_service, "CAPABILITY_PATH", capability_path)

    state = runtime_service.capability_state()

    assert state["available"] is True
    assert state["runtime_id"] == "edge-test-01"
    assert state["runtime_version"] == "2.1.0"
    assert state["capabilities"]["runtime_bridge"] is True


def test_live_detections_preserves_runtime_identity(tmp_path, monkeypatch):
    capability_path = tmp_path / "capability.json"
    live_state_path = tmp_path / "live_state.json"
    capability_path.write_text(
        json.dumps({"runtime_id": "edge-test-02", "runtime_version": "2.1.0", "capabilities": {}}),
        encoding="utf-8",
    )
    live_state_path.write_text(
        json.dumps({"presence": {"registered": [], "unknown": []}, "objects": [], "recent_events": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_service, "CAPABILITY_PATH", capability_path)
    monkeypatch.setattr(runtime_service, "LIVE_STATE_PATH", live_state_path)

    state = runtime_service.live_detections()

    assert state["runtime"] == {
        "id": "edge-test-02",
        "version": "2.1.0",
        "capabilities": {},
    }
    assert state["presence"] == {"registered": [], "unknown": []}
