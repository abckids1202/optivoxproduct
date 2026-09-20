from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment_security import (
    CameraHealthMonitor,
    DeploymentSecurityError,
    build_model_manifest,
    check_required_dependencies,
    safe_local_path,
    validate_camera_source,
    validate_control_plane_url,
    validate_edge_configuration,
    validate_webhook_url,
    verify_model_manifest,
)
from supply_chain_scan import scan


def test_tampered_model_is_rejected(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"trusted-model")
    manifest_path = tmp_path / "models" / "model_checksums.json"
    build_model_manifest([model], tmp_path, manifest_path)
    model.write_bytes(b"modified-model")
    result = verify_model_manifest([model], tmp_path, manifest_path)
    assert result["status"] == "FAILED"
    assert "checksum mismatch" in result["issues"][0]
    with pytest.raises(DeploymentSecurityError):
        verify_model_manifest([model], tmp_path, manifest_path, required=True)


def test_model_manifest_rejects_traversal_entry(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": {"../model.onnx": {"sha256": "0" * 64}}}), encoding="utf-8")
    result = verify_model_manifest([model], tmp_path, manifest)
    assert result["status"] == "FAILED"
    assert "not listed" in result["issues"][0]


def test_unsafe_webhook_urls_are_rejected():
    with pytest.raises(DeploymentSecurityError):
        validate_webhook_url("http://example.com/hook", "production", ["example.com"])
    with pytest.raises(DeploymentSecurityError):
        validate_webhook_url("https://user:password@example.com/hook", "production", ["example.com"])
    with pytest.raises(DeploymentSecurityError):
        validate_webhook_url("https://example.com/hook?token=secret", "production", ["example.com"])
    with pytest.raises(DeploymentSecurityError):
        validate_webhook_url("https://example.com:bad/hook", "production", ["example.com"])
    assert validate_webhook_url("https://alerts.example.com/hook", "production", ["example.com"])["host"] == "alerts.example.com"


def test_control_plane_requires_https_allowlist_and_public_destination():
    with pytest.raises(DeploymentSecurityError):
        validate_control_plane_url("http://control.example.com/ingest", "production", ["control.example.com"])
    with pytest.raises(DeploymentSecurityError):
        validate_control_plane_url("https://control.example.com/ingest?secret=bad", "production", ["control.example.com"])
    with pytest.raises(DeploymentSecurityError):
        validate_control_plane_url("https://127.0.0.1/ingest", "production", ["127.0.0.1"])
    assert validate_control_plane_url("https://control.example.com/ingest", "production", ["example.com"])["host"] == "control.example.com"


def test_invalid_paths_and_camera_sources_fail(tmp_path):
    with pytest.raises(DeploymentSecurityError):
        safe_local_path("../outside.mp4", tmp_path)
    with pytest.raises(DeploymentSecurityError):
        validate_camera_source("file:///etc/passwd", tmp_path)
    with pytest.raises(DeploymentSecurityError):
        validate_camera_source(-1, tmp_path)
    assert validate_camera_source(0, tmp_path)["kind"] == "index"


def test_invalid_strict_configuration_fails_closed(tmp_path):
    config = {
        "CAMERAS": [{"id": "cam-1", "source": "../camera.mp4", "enabled": True}],
        "MODEL_PATH": "missing.onnx",
        "DANGER_DETECTION": {"ENABLED": False},
        "YOLO_CONF": 2,
        "CAPTURE_WIDTH": 1280,
        "CAPTURE_HEIGHT": 720,
        "PROCESSING_WIDTH": 1280,
        "PROCESSING_HEIGHT": 720,
    }
    result = validate_edge_configuration(config, tmp_path, "production")
    assert result["status"] == "INVALID"
    assert result["issues"]
    assert any("model" in issue.lower() for issue in result["issues"])


def test_strict_runtime_cannot_disable_correlation_authority(tmp_path):
    result = validate_edge_configuration({
        "CAMERAS": [{"id": "cam-1", "source": 0, "enabled": True}],
        "MODEL_PATH": None,
        "CORRELATION_CORE": {"ENABLED": False},
    }, tmp_path, "pilot")
    assert result["status"] == "INVALID"
    assert any("correlation_core" in issue.lower() for issue in result["issues"])


def test_invalid_model_registry_is_reported_by_startup_validation(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "model_registry.json").write_text(json.dumps({
        "schema_version": 1,
        "models": [{
            "name": "unsafe", "task": "test", "version": "1",
            "path": "../outside.onnx", "sha256": "0" * 64,
            "source": "test", "license": "test", "evaluation_report": "test",
        }],
    }), encoding="utf-8")
    result = validate_edge_configuration(
        {"CAMERAS": [{"id": "cam-1", "source": 0}], "MODEL_PATH": None},
        tmp_path, "development")
    assert result["model_registry"]["status"] == "INVALID"
    assert any("model registry" in issue for issue in result["issues"])


def test_unknown_runtime_mode_is_invalid(tmp_path):
    config = {"CAMERAS": [{"id": "cam-1", "source": 0}], "MODEL_PATH": None}
    result = validate_edge_configuration(config, tmp_path, "mystery")
    assert result["status"] == "INVALID"
    assert any("runtime mode" in issue for issue in result["issues"])


def test_missing_dependency_is_reported_without_importing_it():
    assert check_required_dependencies(["json"])["status"] == "OK"
    result = check_required_dependencies(["module_that_does_not_exist_optivox"])
    assert result == {"status": "MISSING", "missing": ["module_that_does_not_exist_optivox"]}


def test_camera_disconnect_and_freeze_are_distinguished():
    monitor = CameraHealthMonitor(freeze_after_sec=2, event_cooldown_sec=0)
    assert monitor.observe(False, False, now=0)["state"] == "DISCONNECTED"
    assert monitor.observe(True, True, b"frame", now=1)["state"] == "HEALTHY"
    assert monitor.observe(True, True, b"frame", now=2.5)["state"] == "HEALTHY"
    frozen = monitor.observe(True, True, b"frame", now=3.1)
    assert frozen["state"] == "FROZEN"
    assert monitor.observe(True, False, None, now=4)["state"] == "NO_FRAME"


def test_supply_chain_metadata_is_pinned():
    assert scan(Path(__file__).resolve().parents[2]) == []
