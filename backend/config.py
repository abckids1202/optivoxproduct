from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = (PROJECT_ROOT / os.getenv("OPTIVOX_DATABASE_PATH", "security.db")).resolve()
RUNTIME_DIR = (PROJECT_ROOT / os.getenv("OPTIVOX_RUNTIME_DIR", "runtime")).resolve()
SNAPSHOTS_DIR = (PROJECT_ROOT / "snapshots").resolve()
REPORTS_DIR = (PROJECT_ROOT / "reports").resolve()
EXPORTS_DIR = (PROJECT_ROOT / "exports").resolve()
MODELS_DIR = (PROJECT_ROOT / "models").resolve()
FRONTEND_DIR = (PROJECT_ROOT / "frontend").resolve()
TIMEZONE_NAME = os.getenv("OPTIVOX_TIMEZONE", "Asia/Jakarta")
TIMEZONE = ZoneInfo(TIMEZONE_NAME)
APP_VERSION = "1.0.0-exhibition"
FRONTEND_ORIGINS = [
    os.getenv("OPTIVOX_FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
    "http://localhost:5173",
]
COMMAND_PIN = os.getenv("OPTIVOX_COMMAND_PIN", "")

RUNTIME_DIR.mkdir(exist_ok=True)


def runtime_path(name: str) -> Path:
    return (RUNTIME_DIR / name).resolve()


LIVE_STATE_PATH = runtime_path("live_state.json")
HEARTBEAT_PATH = runtime_path("heartbeat.json")
LATEST_FRAME_PATH = runtime_path("latest_frame.jpg")
COMMANDS_PATH = runtime_path("commands.json")
COMMAND_RESULTS_PATH = runtime_path("command_results.json")
ENROLLMENT_STATUS_PATH = runtime_path("enrollment_status.json")

