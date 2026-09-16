from __future__ import annotations

import os
from pathlib import Path
from datetime import date, datetime
from urllib.parse import urlparse
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
RUNTIME_MODE = os.getenv("OPTIVOX_RUNTIME_MODE", "development").strip().lower()
RUNTIME_MODES = frozenset({"development", "exhibition", "pilot", "production"})
DEMO_MODE = os.getenv("OPTIVOX_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
HOST = os.getenv("OPTIVOX_HOST", "127.0.0.1").strip()
SESSION_COOKIE_NAME = "optivox_session"
CSRF_COOKIE_NAME = "optivox_csrf"
COOKIE_SECURE = os.getenv("OPTIVOX_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"} or RUNTIME_MODE in {"pilot", "production"}
COOKIE_SAMESITE = "strict"
_configured_frontend_origin = os.getenv("OPTIVOX_FRONTEND_ORIGIN", "").strip()
FRONTEND_ORIGINS = ([_configured_frontend_origin] if _configured_frontend_origin else ["http://127.0.0.1:5173", "http://localhost:5173"])
COMMAND_PIN = os.getenv("OPTIVOX_COMMAND_PIN", "")


def frontend_origin_allowed(origin: str | None) -> bool:
    """Validate browser origins for cookie-authenticated cross-origin calls."""
    if not origin:
        return True
    normalized = origin.rstrip("/")
    return normalized in {configured.rstrip("/") for configured in FRONTEND_ORIGINS}


def configured_auth_methods() -> dict[str, bool]:
    """Return whether an operator/admin authentication method is configured.

    Values are deliberately reduced to booleans so diagnostics cannot expose
    credentials or password material.
    """
    operator_key = bool(os.getenv("OPTIVOX_API_KEY") or os.getenv("OPTIVOX_OPERATOR_KEY"))
    admin_key = bool(os.getenv("OPTIVOX_ADMIN_KEY"))
    operator_user = bool(os.getenv("OPTIVOX_OPERATOR_USERNAME") and os.getenv("OPTIVOX_OPERATOR_PASSWORD"))
    admin_user = bool(os.getenv("OPTIVOX_ADMIN_USERNAME") and os.getenv("OPTIVOX_ADMIN_PASSWORD"))
    return {
        "operator_key": operator_key,
        "admin_key": admin_key,
        "operator_user": operator_user,
        "admin_user": admin_user,
    }


def loopback_bypass_allowed() -> bool:
    """Development convenience; never enabled for pilot or production."""
    return RUNTIME_MODE in {"development", "exhibition"}


def configuration_issues() -> list[str]:
    """Return safe-to-display configuration errors for the current runtime mode."""
    issues: list[str] = []
    if RUNTIME_MODE not in RUNTIME_MODES:
        issues.append("OPTIVOX_RUNTIME_MODE must be development, exhibition, pilot, or production.")

    auth = configured_auth_methods()
    if RUNTIME_MODE in {"pilot", "production"} and not any(auth.values()):
        issues.append("pilot and production modes require an API key or configured durable user.")
    if RUNTIME_MODE == "production" and not (auth["admin_key"] or auth["admin_user"]):
        issues.append("production mode requires an administrator API key or administrator user.")
    if RUNTIME_MODE in {"pilot", "production"} and DEMO_MODE:
        issues.append("demo mode must be disabled in pilot and production modes.")

    for origin in FRONTEND_ORIGINS:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or "*" in origin:
            issues.append("OPTIVOX_FRONTEND_ORIGIN must be an explicit http or https origin.")
            break
    if RUNTIME_MODE == "production":
        non_loopback_origins = [origin for origin in FRONTEND_ORIGINS if not any(host in origin for host in ("127.0.0.1", "localhost", "[::1]"))]
        if non_loopback_origins and any(not origin.startswith("https://") for origin in non_loopback_origins):
            issues.append("production frontend origins must use HTTPS when they are not loopback origins.")

    for label, username_var, password_var in (
        ("administrator", "OPTIVOX_ADMIN_USERNAME", "OPTIVOX_ADMIN_PASSWORD"),
        ("operator", "OPTIVOX_OPERATOR_USERNAME", "OPTIVOX_OPERATOR_PASSWORD"),
    ):
        username_set = bool(os.getenv(username_var))
        password_set = bool(os.getenv(password_var))
        if username_set != password_set:
            issues.append(f"{label} username and password must be configured together.")
    return sorted(set(issues))


def validate_runtime_configuration() -> None:
    """Fail closed during pilot/production startup when security is incomplete."""
    issues = configuration_issues()
    if issues:
        raise RuntimeError("OptiVox runtime configuration is unsafe:\n- " + "\n- ".join(issues))


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
