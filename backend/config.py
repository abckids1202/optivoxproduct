from __future__ import annotations

import os
from pathlib import Path
from datetime import date, datetime
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
DEVICE_ID = os.getenv("OPTIVOX_DEVICE_ID", "local-edge-cam-0")
FRONTEND_ORIGINS = [
    os.getenv("OPTIVOX_FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
    "http://localhost:5173",
]
COMMAND_PIN = os.getenv("OPTIVOX_COMMAND_PIN", "")


def _int_list(value: str, default: list[int]) -> list[int]:
    parsed = []
    for item in str(value or "").split(","):
        try:
            number = int(item.strip())
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 6:
            parsed.append(number)
    return parsed or list(default)


SCHOOL_DAYS = _int_list(os.getenv("OPTIVOX_SCHOOL_DAYS", "0,1,2,3,4"), [0, 1, 2, 3, 4])
SCHOOL_HOLIDAYS = [
    item.strip() for item in os.getenv("OPTIVOX_SCHOOL_HOLIDAYS", "").split(",")
    if item.strip()
]

RUNTIME_DIR.mkdir(exist_ok=True)


def runtime_path(name: str) -> Path:
    return (RUNTIME_DIR / name).resolve()


def local_now() -> datetime:
    return datetime.now(TIMEZONE)


def local_today() -> date:
    return local_now().date()


LIVE_STATE_PATH = runtime_path("live_state.json")
HEARTBEAT_PATH = runtime_path("heartbeat.json")
LATEST_FRAME_PATH = runtime_path("latest_frame.jpg")
COMMANDS_PATH = runtime_path("commands.json")
COMMAND_RESULTS_PATH = runtime_path("command_results.json")
ENROLLMENT_STATUS_PATH = runtime_path("enrollment_status.json")
CAPABILITY_PATH = runtime_path("capability.json")
PERFORMANCE_SUMMARY_PATH = runtime_path("performance_summary.json")
