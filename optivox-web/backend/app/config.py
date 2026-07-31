"""OptiVox backend configuration."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _csv(value, default):
    raw = value or default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Config:
    SECRET_KEY = os.environ.get("OPTIVOX_SECRET_KEY", "change-me-in-production")
    DEBUG = _bool(os.environ.get("OPTIVOX_DEBUG"), False)
    BRAND = "OptiVox"

    ORG_USERNAME = os.environ.get("OPTIVOX_USER", "admin")
    ORG_PASSWORD = os.environ.get("OPTIVOX_PASS", "optivox")

    DATABASE_FILE = os.environ.get("OPTIVOX_DB", str(PROJECT_DIR / "security.db"))
    VISION_MODULE = os.environ.get("OPTIVOX_VISION_MODULE", "optivox_core")
    CAMERA_SOURCE = os.environ.get("OPTIVOX_CAMERA", "0")
    CAMERA_ID = os.environ.get("OPTIVOX_CAMERA_ID", "cam_0")
    CAMERA_LOCATION = os.environ.get("OPTIVOX_CAMERA_LOCATION", "Main Entrance")

    ENABLE_BRIDGE = _bool(os.environ.get("OPTIVOX_ENABLE_BRIDGE"), True)
    ALLOW_SYNTHETIC = _bool(os.environ.get("OPTIVOX_SYNTHETIC"), True)
    JPEG_QUALITY = int(os.environ.get("OPTIVOX_JPEG_QUALITY", "72"))
    STREAM_FPS = int(os.environ.get("OPTIVOX_STREAM_FPS", "20"))

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.environ.get("OPTIVOX_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = _bool(os.environ.get("OPTIVOX_COOKIE_SECURE"), False)

    CORS_ORIGINS = _csv(
        os.environ.get("OPTIVOX_FRONTEND_URL"),
        "http://localhost:5173,http://127.0.0.1:5173",
    )
