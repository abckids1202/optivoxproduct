"""Small, dependency-free controls for the OptiVox edge deployment boundary.

This module deliberately does not import OpenCV, Ultralytics, or InsightFace.
It is safe to use during startup and in tests before expensive model loading.
Production and pilot modes fail closed; development and exhibition modes keep
local compatibility but report missing controls explicitly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlparse


MODEL_SUFFIXES = frozenset({".onnx", ".pt", ".pth", ".engine", ".bin"})
STRICT_MODES = frozenset({"pilot", "production"})
RUNTIME_MODES = frozenset({"development", "exhibition", "pilot", "production"})
SAFE_CAMERA_URL_SCHEMES = frozenset({"rtsp", "rtsps"})


class DeploymentSecurityError(RuntimeError):
    """Raised when a deployment boundary cannot be trusted."""


def is_strict_mode(mode: str | None) -> bool:
    return str(mode or os.getenv("OPTIVOX_RUNTIME_MODE", "development")).strip().lower() in STRICT_MODES


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_local_path(value: str | os.PathLike[str], root: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Resolve a local path and reject traversal or NUL characters."""
    raw = os.fspath(value)
    if "\x00" in raw:
        raise DeploymentSecurityError("path contains a NUL character")
    root_path = Path(root).resolve()
    path = Path(raw)
    resolved = (path if path.is_absolute() else root_path / path).resolve()
    if not _inside(resolved, root_path):
        raise DeploymentSecurityError(f"path escapes its allowed root: {raw}")
    if must_exist and not resolved.exists():
        raise DeploymentSecurityError(f"path does not exist: {raw}")
    return resolved


def validate_camera_source(source: Any, root: str | os.PathLike[str], mode: str = "development") -> dict[str, Any]:
    """Validate a camera source before passing it to cv2.VideoCapture."""
    if isinstance(source, bool):
        raise DeploymentSecurityError("camera source must not be boolean")
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        index = int(source)
        if index < 0:
            raise DeploymentSecurityError("camera index must be non-negative")
        return {"kind": "index", "value": index}
    if not isinstance(source, (str, os.PathLike)):
        raise DeploymentSecurityError("camera source must be an index, local file, or RTSP URL")
    raw = os.fspath(source).strip()
    parsed = urlparse(raw)
    if parsed.scheme:
        if parsed.scheme.lower() not in SAFE_CAMERA_URL_SCHEMES:
            raise DeploymentSecurityError("camera URL scheme must be rtsp or rtsps")
        if not parsed.hostname or parsed.fragment:
            raise DeploymentSecurityError("camera URL must contain a host and no fragment")
        if parsed.username is not None or parsed.password is not None:
            raise DeploymentSecurityError("camera URL credentials must be supplied through protected configuration")
        return {"kind": "url", "scheme": parsed.scheme.lower(), "host": parsed.hostname}
    path = safe_local_path(raw, root, must_exist=is_strict_mode(mode))
    if path.is_dir():
        raise DeploymentSecurityError("camera source cannot be a directory")
    return {"kind": "file", "value": str(path)}


def _host_allowed(host: str, allowed_hosts: Iterable[str]) -> bool:
    normalized = host.casefold().rstrip(".")
    for allowed in allowed_hosts:
        candidate = str(allowed).strip().casefold().rstrip(".")
        if candidate and (normalized == candidate or normalized.endswith("." + candidate)):
            return True
    return False


def validate_webhook_url(url: str, mode: str = "development", allowed_hosts: Iterable[str] = ()) -> dict[str, Any]:
    """Allow only explicit, credential-free webhook destinations."""
    if not isinstance(url, str) or not url.strip():
        raise DeploymentSecurityError("webhook URL is empty")
    parsed = urlparse(url.strip())
    host = parsed.hostname
    if not host or parsed.username is not None or parsed.password is not None:
        raise DeploymentSecurityError("webhook URL must have a host and no embedded credentials")
    if parsed.fragment or parsed.query:
        raise DeploymentSecurityError("webhook URL must not contain query strings or fragments")
    strict = is_strict_mode(mode)
    if parsed.scheme.lower() != "https":
        if not (not strict and parsed.scheme.lower() == "http" and host.casefold() in {"localhost", "127.0.0.1", "::1"}):
            raise DeploymentSecurityError("webhook URL must use HTTPS outside local development")
    configured_hosts = tuple(str(item).strip() for item in allowed_hosts if str(item).strip())
    if strict and not configured_hosts:
        raise DeploymentSecurityError("strict webhook delivery requires OPTIVOX_ALLOWED_WEBHOOK_HOSTS")
    if configured_hosts and not _host_allowed(host, configured_hosts):
        raise DeploymentSecurityError("webhook host is not in the outbound host allowlist")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if strict and ip is not None and (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified):
        raise DeploymentSecurityError("strict webhook delivery cannot target a private or loopback IP")
    try:
        port = parsed.port
    except ValueError as exc:
        raise DeploymentSecurityError("webhook URL port is invalid") from exc
    return {"scheme": parsed.scheme.lower(), "host": host.casefold(), "port": port}


def validate_alert_config(alert_config: Mapping[str, Any], mode: str = "development") -> list[str]:
    """Return safe configuration issues without exposing credential values."""
    issues: list[str] = []
    webhook = alert_config.get("webhook", {}) if isinstance(alert_config, Mapping) else {}
    if webhook.get("enabled"):
        try:
            allowed = os.getenv("OPTIVOX_ALLOWED_WEBHOOK_HOSTS", "").split(",")
            validate_webhook_url(str(webhook.get("url", "")), mode, allowed)
        except DeploymentSecurityError as exc:
            issues.append(str(exc))
    for key in ("smtp_server",):
        value = alert_config.get("email", {}).get(key) if isinstance(alert_config.get("email", {}), Mapping) else None
        if value and ("/" in str(value) or "\\" in str(value) or " " in str(value)):
            issues.append("SMTP server must be a hostname")
    return sorted(set(issues))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_manifest(model_paths: Iterable[str | os.PathLike[str]], root: str | os.PathLike[str], output: str | os.PathLike[str]) -> dict[str, Any]:
    """Create a checksum manifest for configured model files."""
    root_path = Path(root).resolve()
    files: dict[str, dict[str, Any]] = {}
    for raw_path in model_paths:
        path = safe_local_path(raw_path, root_path, must_exist=True)
        if path.suffix.lower() not in MODEL_SUFFIXES:
            continue
        relative = path.relative_to(root_path).as_posix()
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    destination = safe_local_path(output, root_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return manifest


def verify_model_manifest(model_paths: Iterable[str | os.PathLike[str]], root: str | os.PathLike[str], manifest_path: str | os.PathLike[str], *, required: bool = False) -> dict[str, Any]:
    """Verify configured model files and reject modified or unlisted files."""
    root_path = Path(root).resolve()
    manifest_file = safe_local_path(manifest_path, root_path)
    if not manifest_file.exists():
        result = {"status": "NOT_CONFIGURED", "verified": [], "issues": ["model checksum manifest is missing"]}
        if required:
            raise DeploymentSecurityError(result["issues"][0])
        return result
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        entries = manifest["files"]
        if manifest.get("schema_version") != 1 or not isinstance(entries, dict):
            raise ValueError("unsupported model manifest")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        if required:
            raise DeploymentSecurityError(f"invalid model checksum manifest: {exc}") from exc
        return {"status": "INVALID", "verified": [], "issues": ["invalid model checksum manifest"]}
    verified: list[str] = []
    issues: list[str] = []
    for raw_path in model_paths:
        try:
            path = safe_local_path(raw_path, root_path)
            relative = path.relative_to(root_path).as_posix()
            if relative not in entries:
                raise DeploymentSecurityError(f"model is not listed in checksum manifest: {relative}")
            if not path.exists():
                raise DeploymentSecurityError(f"model file is missing: {relative}")
            expected = entries[relative].get("sha256")
            actual = sha256_file(path)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected or "")) or actual.casefold() != str(expected).casefold():
                raise DeploymentSecurityError(f"model checksum mismatch: {relative}")
            verified.append(relative)
        except (OSError, DeploymentSecurityError) as exc:
            issues.append(str(exc))
    result = {"status": "VERIFIED" if not issues else "FAILED", "verified": verified, "issues": issues}
    if required and issues:
        raise DeploymentSecurityError("model verification failed: " + "; ".join(issues))
    return result


def verify_face_model_cache(name: str = "buffalo_l", root: str | os.PathLike[str] | None = None, *, manifest_path: str | os.PathLike[str] | None = None, required: bool = False) -> dict[str, Any]:
    """Verify the local InsightFace cache before FaceAnalysis can download."""
    cache_root = Path(root or os.getenv("OPTIVOX_INSIGHTFACE_ROOT", str(Path.home() / ".insightface"))).expanduser().resolve()
    model_dir = cache_root / "models" / str(name)
    files = sorted(path for path in model_dir.glob("*.onnx") if path.is_file()) if model_dir.exists() else []
    manifest = Path(manifest_path or os.getenv("OPTIVOX_FACE_MODEL_MANIFEST", str(model_dir / "model_checksums.json"))).expanduser()
    if not files:
        result = {"status": "MISSING", "verified": [], "issues": [f"InsightFace cache is missing: {model_dir}"]}
        if required:
            raise DeploymentSecurityError(result["issues"][0])
        return result
    return verify_model_manifest(files, model_dir, manifest, required=required)


def check_required_dependencies(requirements: Iterable[str]) -> dict[str, Any]:
    """Check import availability without importing untrusted packages."""
    missing = []
    for module_name in requirements:
        if not importlib.util.find_spec(str(module_name)):
            missing.append(str(module_name))
    return {"status": "OK" if not missing else "MISSING", "missing": missing}


def validate_edge_configuration(config: Mapping[str, Any], root: str | os.PathLike[str], mode: str, *, manifest_path: str | os.PathLike[str] = "models/model_checksums.json") -> dict[str, Any]:
    """Validate camera, model, and numeric runtime settings before startup."""
    issues: list[str] = []
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in RUNTIME_MODES:
        issues.append("runtime mode must be development, exhibition, pilot, or production")
    cameras = config.get("CAMERAS") or []
    enabled = [item for item in cameras if isinstance(item, Mapping) and item.get("enabled", True)]
    if not enabled:
        issues.append("at least one enabled camera is required")
    seen_ids = set()
    for camera in enabled:
        camera_id = str(camera.get("id", "")).strip()
        if not camera_id or camera_id in seen_ids:
            issues.append("camera IDs must be non-empty and unique")
        seen_ids.add(camera_id)
        try:
            validate_camera_source(camera.get("source"), root, mode)
        except DeploymentSecurityError as exc:
            issues.append(f"camera {camera_id or '<unknown>'}: {exc}")
    model_paths = [config.get("MODEL_PATH")]
    danger = config.get("DANGER_DETECTION") or {}
    if danger.get("ENABLED"):
        model_paths += list(danger.get("MODEL_PATHS") or [])
    model_paths = [path for path in model_paths if path]
    try:
        model_result = verify_model_manifest(model_paths, root, manifest_path, required=is_strict_mode(mode))
    except DeploymentSecurityError as exc:
        model_result = {"status": "FAILED", "verified": [], "issues": [str(exc)]}
        issues.extend(model_result["issues"])
    try:
        face_model_result = verify_face_model_cache(required=is_strict_mode(mode))
    except DeploymentSecurityError as exc:
        face_model_result = {"status": "FAILED", "verified": [], "issues": [str(exc)]}
        issues.extend(face_model_result["issues"])
    if not 0 < float(config.get("YOLO_CONF", 0.4)) <= 1:
        issues.append("YOLO_CONF must be between 0 and 1")
    for key in ("CAPTURE_WIDTH", "CAPTURE_HEIGHT", "PROCESSING_WIDTH", "PROCESSING_HEIGHT"):
        try:
            if int(config.get(key, 0)) <= 0:
                issues.append(f"{key} must be positive")
        except (TypeError, ValueError):
            issues.append(f"{key} must be an integer")
    return {"status": "VALID" if not issues else "INVALID", "issues": sorted(set(issues), key=str), "models": model_result, "face_model": face_model_result}


def process_identity() -> dict[str, Any]:
    """Return non-secret identity information for health and watchdog records."""
    parent_pid = os.getppid()
    return {
        "pid": os.getpid(),
        "parent_pid": parent_pid,
        "executable": str(Path(sys.executable).resolve()),
        "command": Path(sys.argv[0]).name if sys.argv else "unknown",
    }


def watchdog_status(expected_pid: int | None = None) -> dict[str, Any]:
    pid = int(expected_pid or os.getpid())
    alive = False
    try:
        os.kill(pid, 0)
        alive = True
    except (OSError, ProcessLookupError):
        alive = False
    return {"pid": pid, "pid_alive": alive, "current_process": pid == os.getpid(), "checked_at": time.time()}


class CameraHealthMonitor:
    """Detect camera disconnects, missing frames, and repeated frozen frames."""

    def __init__(self, freeze_after_sec: float = 5.0, event_cooldown_sec: float = 10.0):
        self.freeze_after_sec = max(0.5, float(freeze_after_sec))
        self.event_cooldown_sec = max(0.0, float(event_cooldown_sec))
        self.state = "STARTING"
        self.last_change_at = time.monotonic()
        self.last_event_at = 0.0
        self.last_frame_hash: Optional[str] = None
        self.same_frame_since: Optional[float] = None
        self.frames_ok = 0
        self.failures = 0

    @staticmethod
    def _frame_hash(frame: Any) -> Optional[str]:
        if frame is None:
            return None
        try:
            return hashlib.sha256(memoryview(frame).tobytes()).hexdigest()
        except (AttributeError, TypeError, ValueError):
            return None

    def observe(self, capture_open: bool, read_ok: bool, frame: Any = None, now: float | None = None) -> dict[str, Any]:
        current = time.monotonic() if now is None else float(now)
        if not capture_open:
            next_state = "DISCONNECTED"
            self.failures += 1
        elif not read_ok or frame is None:
            next_state = "NO_FRAME"
            self.failures += 1
        else:
            self.frames_ok += 1
            fingerprint = self._frame_hash(frame)
            if fingerprint and fingerprint == self.last_frame_hash:
                self.same_frame_since = self.same_frame_since if self.same_frame_since is not None else current
            else:
                self.same_frame_since = current
            self.last_frame_hash = fingerprint
            next_state = "FROZEN" if self.same_frame_since is not None and current - self.same_frame_since >= self.freeze_after_sec else "HEALTHY"
        changed = next_state != self.state
        if changed:
            self.last_change_at = current
            self.state = next_state
        event = None
        if changed:
            self.last_event_at = current
            event = {"type": "CAMERA_HEALTH", "state": self.state, "timestamp": datetime.now(timezone.utc).isoformat()}
        return {"state": self.state, "changed": changed, "event": event, "frames_ok": self.frames_ok, "failures": self.failures, "same_frame_seconds": round(max(0.0, current - (self.same_frame_since or current)), 3)}


def append_runtime_health_event(runtime_dir: str | os.PathLike[str], event: Mapping[str, Any]) -> None:
    path = safe_local_path("health_events.jsonl", runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(), **dict(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
