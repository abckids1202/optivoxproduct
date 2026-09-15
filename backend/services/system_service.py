from __future__ import annotations

import json
import importlib.util
import os
import shutil

from ..config import DATABASE_PATH, EXPORTS_DIR, MODELS_DIR, REPORTS_DIR, SNAPSHOTS_DIR
from ..database import table_counts
from .runtime_service import heartbeat_state, live_state, performance_report


def system_status() -> dict:
    live = live_state()
    hb = heartbeat_state()
    return {
        "engine": {"state": hb["status"], **live.get("engine", {})},
        "camera": {"state": live.get("engine", {}).get("camera"), "location": live.get("engine", {}).get("location")},
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
    }


def database_status() -> dict:
    return {
        "connected": DATABASE_PATH.exists(),
        "path": str(DATABASE_PATH),
        "size_bytes": DATABASE_PATH.stat().st_size if DATABASE_PATH.exists() else 0,
        "tables": table_counts() if DATABASE_PATH.exists() else {},
    }


def storage_status() -> dict:
    usage = shutil.disk_usage(SNAPSHOTS_DIR if SNAPSHOTS_DIR.exists() else DATABASE_PATH.parent)
    return {
        "snapshots": count_and_size(SNAPSHOTS_DIR),
        "reports": count_and_size(REPORTS_DIR),
        "exports": count_and_size(EXPORTS_DIR),
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
