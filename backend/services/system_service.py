from __future__ import annotations

import json
import importlib.util
import os
import shutil

from ..config import DATABASE_PATH, EXPORTS_DIR, MODELS_DIR, REPORTS_DIR, SECURITY_ZONES_PATH, SNAPSHOTS_DIR
from ..database import database_health, table_counts
from core.model_registry import registry_snapshot
from .runtime_service import heartbeat_state, live_state, performance_report
from .outbox_service import summary as outbox_summary
from .storage_service import backup_recovery_status, list_backups
from .backup_scheduler import scheduler_status


def system_status() -> dict:
    live = live_state()
    hb = heartbeat_state()
    return {
        "engine": {"state": hb["status"], **live.get("engine", {})},
        "camera": {
            "state": (live.get("camera") or {}).get("status") or hb.get("heartbeat", {}).get("camera_status"),
            "id": (live.get("camera") or {}).get("id"),
            "location": (live.get("camera") or {}).get("location"),
        },
        "database": database_status(),
        "storage": storage_status(),
    }


def models_status() -> dict:
    object_model = next(
        (path for path in (MODELS_DIR / "yolov8n.pt", MODELS_DIR.parent / "yolov8n.pt") if path.exists()),
        None,
    )
    danger_model = next(
        (path for path in (
            MODELS_DIR / "best_weapon.onnx",
            MODELS_DIR / "best_weapon.pt",
        ) if path.exists()),
        None,
    )
    registry = registry_snapshot(MODELS_DIR.parent, MODELS_DIR / "model_registry.json")
    return {
        # These are capability probes, not claims that a live runtime is
        # currently serving inference. The live handshake remains the source
        # of truth for runtime availability.
        "face_detector": "installed" if importlib.util.find_spec("insightface") else "missing",
        "face_recognizer": "installed" if importlib.util.find_spec("insightface") else "missing",
        "faiss": "installed" if importlib.util.find_spec("faiss") else "numpy_fallback",
        "object_detector": "configured" if object_model else "missing",
        "object_model_path": str(object_model) if object_model else None,
        "pose_detector": "installed" if importlib.util.find_spec("mediapipe") else "missing",
        "hand_detector": "installed" if importlib.util.find_spec("mediapipe") else "missing",
        "anti_spoofing": "experimental_heuristics",
        "danger_model": "configured" if danger_model else "not_configured",
        "danger_model_path": str(danger_model) if danger_model else None,
        "ai_assistant": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured",
        "registry": registry,
    }


def database_status() -> dict:
    health = database_health(include_counts=False)
    # The database module keeps its absolute path for local diagnostics, but
    # public API responses must not disclose workstation/project layout.
    public_health = dict(health)
    public_health.pop("path", None)
    counts = {}
    if DATABASE_PATH.exists() and health.get("connected"):
        try:
            counts = table_counts()
        except Exception:
            # Keep the health endpoint usable when the database is damaged;
            # the integrity report is the authoritative failure signal.
            counts = {}
    try:
        outbox = outbox_summary() if health.get("connected") else {"status": "unavailable"}
    except Exception:
        # A legacy or partially migrated database must still expose its
        # integrity report instead of failing the whole system-status route.
        outbox = {"status": "unavailable", "reason": "schema_not_ready"}
    return {
        "connected": DATABASE_PATH.exists(),
        "size_bytes": DATABASE_PATH.stat().st_size if DATABASE_PATH.exists() else 0,
        "tables": counts,
        "health": public_health,
        "outbox": outbox,
        "backup_recovery": backup_recovery_status(),
    }


def storage_status() -> dict:
    usage = shutil.disk_usage(SNAPSHOTS_DIR if SNAPSHOTS_DIR.exists() else DATABASE_PATH.parent)
    backups = list_backups()
    return {
        "snapshots": count_and_size(SNAPSHOTS_DIR),
        "reports": count_and_size(REPORTS_DIR),
        "exports": count_and_size(EXPORTS_DIR),
        "backups": backups,
        "backup_scheduler": scheduler_status(),
        "backup_recovery": backup_recovery_status(),
        "free_bytes": usage.free,
    }


def count_and_size(path):
    if not path.exists():
        return {"count": 0, "size_bytes": 0}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"count": len(files), "size_bytes": sum(p.stat().st_size for p in files)}


def alerts_status() -> dict:
    config_path = DATABASE_PATH.parent / "alert_config.json"
    data = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return {
        "email_enabled": bool(data.get("email", {}).get("enabled")),
        "telegram_enabled": bool(data.get("telegram", {}).get("enabled")),
        "webhook_enabled": bool(data.get("webhook", {}).get("enabled")),
    }


def security_zones_status() -> dict:
    """Expose normalized policy metadata without exposing biometric material."""
    if not SECURITY_ZONES_PATH.exists():
        return {"source": "startup_config", "zones": [], "updated_at": None}
    try:
        data = json.loads(SECURITY_ZONES_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "zones" not in data:
            raise ValueError("security-zone override must contain a zones list")
        zones = data.get("zones")
        if not isinstance(zones, list):
            raise ValueError("zones must be a list")
        return {
            "source": "runtime_override",
            "zones": zones,
            "updated_at": data.get("updated_at") if isinstance(data, dict) else None,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"source": "invalid_override", "zones": [], "updated_at": None}
