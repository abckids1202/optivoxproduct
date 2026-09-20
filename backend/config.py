from __future__ import annotations

import json
import importlib.util
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
BACKUPS_DIR = (PROJECT_ROOT / os.getenv("OPTIVOX_BACKUP_DIR", "backups")).resolve()
MODELS_DIR = (PROJECT_ROOT / "models").resolve()
FRONTEND_DIR = (PROJECT_ROOT / "frontend").resolve()
TIMEZONE_NAME = os.getenv("OPTIVOX_TIMEZONE", "Asia/Jakarta")
TIMEZONE = ZoneInfo(TIMEZONE_NAME)
APP_VERSION = "1.0.0-exhibition"
DEVICE_ID = os.getenv("OPTIVOX_DEVICE_ID", "local-edge-cam-0")
SITE_ID = os.getenv("OPTIVOX_SITE_ID", "local-site")
ORGANIZATION_ID = os.getenv("OPTIVOX_ORGANIZATION_ID", "local-organization")
RUNTIME_MODE = os.getenv("OPTIVOX_RUNTIME_MODE", "development").strip().lower()
RUNTIME_MODES = frozenset({"development", "exhibition", "pilot", "production"})
SQLITE_SYNCHRONOUS = os.getenv(
    "OPTIVOX_SQLITE_SYNCHRONOUS",
    "FULL" if RUNTIME_MODE in {"pilot", "production"} else "NORMAL",
).strip().upper()
SQLITE_SECURE_DELETE = os.getenv(
    "OPTIVOX_SQLITE_SECURE_DELETE",
    "ON" if RUNTIME_MODE in {"pilot", "production"} else "OFF",
).strip().upper()
DEMO_MODE = os.getenv("OPTIVOX_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
HOST = os.getenv("OPTIVOX_HOST", "127.0.0.1").strip()
SESSION_COOKIE_NAME = "optivox_session"
CSRF_COOKIE_NAME = "optivox_csrf"
COOKIE_SECURE = os.getenv("OPTIVOX_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"} or RUNTIME_MODE in {"pilot", "production"}
COOKIE_SAMESITE = "strict"
_configured_frontend_origin = os.getenv("OPTIVOX_FRONTEND_ORIGIN", "").strip()
FRONTEND_ORIGINS = ([_configured_frontend_origin] if _configured_frontend_origin else ["http://127.0.0.1:5173", "http://localhost:5173"])
COMMAND_PIN = os.getenv("OPTIVOX_COMMAND_PIN", "")


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


DATABASE_MAX_BYTES = _positive_int_env("OPTIVOX_DATABASE_MAX_BYTES", 4 * 1024 * 1024 * 1024)
DATABASE_WARN_BYTES = _positive_int_env("OPTIVOX_DATABASE_WARN_BYTES", 3 * 1024 * 1024 * 1024)
DATABASE_WAL_MAX_BYTES = _positive_int_env("OPTIVOX_DATABASE_WAL_MAX_BYTES", 512 * 1024 * 1024)
DATABASE_MIN_FREE_BYTES = _positive_int_env("OPTIVOX_DATABASE_MIN_FREE_BYTES", 256 * 1024 * 1024)
DATABASE_ENCRYPTION_ENABLED = os.getenv(
    "OPTIVOX_DATABASE_ENCRYPTION_ENABLED",
    "true" if RUNTIME_MODE in {"pilot", "production"} else "false",
).strip().lower() in {"1", "true", "yes", "on"}
DATABASE_ENCRYPTION_KEY = os.getenv("OPTIVOX_DATABASE_ENCRYPTION_KEY", "").strip()
DATABASE_ENCRYPTION_DRIVER = os.getenv("OPTIVOX_DATABASE_ENCRYPTION_DRIVER", "sqlcipher").strip().lower() or "sqlcipher"
EVIDENCE_RETENTION_DAYS = _positive_int_env("OPTIVOX_EVIDENCE_RETENTION_DAYS", 30)
EVIDENCE_ENCRYPTION_KEY = os.getenv("OPTIVOX_EVIDENCE_ENCRYPTION_KEY", "").strip()
EVIDENCE_ENCRYPTION_KEY_ID = os.getenv("OPTIVOX_EVIDENCE_ENCRYPTION_KEY_ID", "primary").strip() or "primary"
EVIDENCE_ENCRYPTION_KEYS_JSON = os.getenv("OPTIVOX_EVIDENCE_ENCRYPTION_KEYS_JSON", "").strip()
EVIDENCE_ENCRYPTION_ENABLED = os.getenv(
    "OPTIVOX_EVIDENCE_ENCRYPTION_ENABLED",
    "true" if RUNTIME_MODE in {"pilot", "production"} else "false",
).strip().lower() in {"1", "true", "yes", "on"}
MAX_REQUEST_BODY_BYTES = _positive_int_env("OPTIVOX_MAX_REQUEST_BODY_BYTES", 1_048_576)
BACKUP_SIGNING_KEY = os.getenv("OPTIVOX_BACKUP_SIGNING_KEY", "").strip()
BACKUP_ENCRYPTION_KEY = os.getenv("OPTIVOX_BACKUP_ENCRYPTION_KEY", "").strip()
BACKUP_ENCRYPTION_KEY_ID = os.getenv("OPTIVOX_BACKUP_ENCRYPTION_KEY_ID", "primary").strip() or "primary"
BACKUP_ENCRYPTION_KEYS_JSON = os.getenv("OPTIVOX_BACKUP_ENCRYPTION_KEYS_JSON", "").strip()
BACKUP_ENCRYPTION_ENABLED = os.getenv(
    "OPTIVOX_BACKUP_ENCRYPTION_ENABLED",
    "true" if RUNTIME_MODE in {"pilot", "production"} else "false",
).strip().lower() in {"1", "true", "yes", "on"}
AUDIT_SIGNING_KEY = os.getenv("OPTIVOX_AUDIT_SIGNING_KEY", "").strip()
BACKUP_SCHEDULER_ENABLED = os.getenv("OPTIVOX_BACKUP_SCHEDULER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
EXTERNAL_BACKUP_CONFIRMED = os.getenv("OPTIVOX_EXTERNAL_BACKUP_CONFIRMED", "false").strip().lower() in {"1", "true", "yes", "on"}
BACKUP_INTERVAL_MINUTES = _positive_int_env("OPTIVOX_BACKUP_INTERVAL_MINUTES", 60)
BACKUP_MAX_AGE_MINUTES = _positive_int_env("OPTIVOX_BACKUP_MAX_AGE_MINUTES", 180)
BACKUP_RETENTION_COUNT = _positive_int_env("OPTIVOX_BACKUP_RETENTION_COUNT", 24)
BACKUP_RETENTION_DAYS = _positive_int_env("OPTIVOX_BACKUP_RETENTION_DAYS", 30)
SYNC_SECRET = os.getenv("OPTIVOX_SYNC_SECRET", "").strip()
SYNC_ENDPOINT = os.getenv("OPTIVOX_SYNC_ENDPOINT", "").strip()
SYNC_ENABLED = os.getenv("OPTIVOX_SYNC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
SYNC_BATCH_SIZE = _positive_int_env("OPTIVOX_SYNC_BATCH_SIZE", 50)
SYNC_TIMEOUT_SECONDS = max(1, min(30, _positive_int_env("OPTIVOX_SYNC_TIMEOUT_SECONDS", 10)))
SYNC_ALLOWED_HOSTS = tuple(
    item.strip() for item in os.getenv("OPTIVOX_SYNC_ALLOWED_HOSTS", "").split(",") if item.strip()
)
SYNC_DEVICE_REGISTRY_JSON = os.getenv("OPTIVOX_SYNC_DEVICE_REGISTRY_JSON", "").strip()
SYNC_PRIVATE_KEY_PATH = os.getenv("OPTIVOX_SYNC_PRIVATE_KEY_PATH", "").strip()
SYNC_KEY_ID = os.getenv("OPTIVOX_SYNC_KEY_ID", "").strip()


def sync_device_registry_issues() -> list[str]:
    """Validate the optional per-device receiver registry without exposing keys."""
    if not SYNC_DEVICE_REGISTRY_JSON:
        return []
    try:
        value = json.loads(SYNC_DEVICE_REGISTRY_JSON)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ["OPTIVOX_SYNC_DEVICE_REGISTRY_JSON must be valid JSON."]
    if not isinstance(value, dict):
        return ["OPTIVOX_SYNC_DEVICE_REGISTRY_JSON must be an object keyed by device ID."]
    issues: list[str] = []
    for device_id, entry in value.items():
        if not isinstance(entry, dict):
            issues.append(f"sync device {device_id} must be an object.")
            continue
        if not str(entry.get("secret") or "").strip() or len(str(entry.get("secret") or "").strip()) < 32:
            issues.append(f"sync device {device_id} requires a secret with at least 32 characters.")
        if not str(entry.get("site_id") or "").strip() or not str(entry.get("organization_id") or "").strip():
            issues.append(f"sync device {device_id} requires site_id and organization_id.")
        if str(entry.get("status", "active")).lower() not in {"active", "revoked", "disabled"}:
            issues.append(f"sync device {device_id} has an invalid status.")
        algorithm = str(entry.get("signature_algorithm", "hmac-sha256") or "hmac-sha256").strip().lower()
        if algorithm not in {"hmac-sha256", "ed25519"}:
            issues.append(f"sync device {device_id} has an unsupported signature algorithm.")
        if algorithm == "ed25519":
            if not str(entry.get("key_id") or "").strip():
                issues.append(f"sync device {device_id} requires key_id for Ed25519 authentication.")
            public_key = str(entry.get("public_key") or "").strip()
            public_keys = entry.get("public_keys")
            if public_keys is not None and not isinstance(public_keys, dict):
                issues.append(f"sync device {device_id} public_keys must be an object keyed by key ID.")
                public_keys = {}
            key_material = dict(public_keys or {})
            if public_key:
                key_material.setdefault(str(entry.get("key_id") or "").strip(), public_key)
            if not key_material:
                issues.append(f"sync device {device_id} requires public_key or public_keys for Ed25519 authentication.")
            if len(key_material) > 4:
                issues.append(f"sync device {device_id} may register at most four Ed25519 public keys.")
            for key_id, configured_key in key_material.items():
                if not str(key_id or "").strip() or not str(configured_key or "").strip():
                    issues.append(f"sync device {device_id} has an empty Ed25519 key entry.")
                    continue
                try:
                    from core.device_auth import load_ed25519_public_key
                    load_ed25519_public_key(configured_key)
                except Exception:
                    issues.append(f"sync device {device_id} has an invalid Ed25519 public key.")
    return sorted(set(issues))


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
    if SQLITE_SYNCHRONOUS not in {"FULL", "NORMAL", "OFF"}:
        issues.append("OPTIVOX_SQLITE_SYNCHRONOUS must be FULL, NORMAL, or OFF.")
    if SQLITE_SECURE_DELETE not in {"ON", "OFF", "FAST"}:
        issues.append("OPTIVOX_SQLITE_SECURE_DELETE must be ON, OFF, or FAST.")
    if DATABASE_WARN_BYTES > DATABASE_MAX_BYTES:
        issues.append("OPTIVOX_DATABASE_WARN_BYTES must not exceed OPTIVOX_DATABASE_MAX_BYTES.")
    if DATABASE_MAX_BYTES <= 0 or DATABASE_WAL_MAX_BYTES <= 0 or DATABASE_MIN_FREE_BYTES <= 0:
        issues.append("database capacity limits must be positive.")
    if DATABASE_ENCRYPTION_DRIVER not in {"sqlcipher", "pysqlcipher3", "sqlcipher3"}:
        issues.append("OPTIVOX_DATABASE_ENCRYPTION_DRIVER must be sqlcipher-compatible.")
    if RUNTIME_MODE in {"pilot", "production"} and not DATABASE_ENCRYPTION_ENABLED:
        issues.append("pilot and production modes require OPTIVOX_DATABASE_ENCRYPTION_ENABLED=true.")
    if DATABASE_ENCRYPTION_ENABLED:
        if len(DATABASE_ENCRYPTION_KEY) < 32:
            issues.append("encrypted databases require OPTIVOX_DATABASE_ENCRYPTION_KEY with at least 32 characters.")
        if not (importlib.util.find_spec("pysqlcipher3") or importlib.util.find_spec("sqlcipher3")):
            issues.append("encrypted databases require an installed SQLCipher-compatible Python driver.")
    if RUNTIME_MODE in {"pilot", "production"} and SQLITE_SYNCHRONOUS != "FULL":
        issues.append("pilot and production modes require OPTIVOX_SQLITE_SYNCHRONOUS=FULL.")
    if RUNTIME_MODE in {"pilot", "production"} and SQLITE_SECURE_DELETE != "ON":
        issues.append("pilot and production modes require OPTIVOX_SQLITE_SECURE_DELETE=ON.")

    auth = configured_auth_methods()
    if RUNTIME_MODE in {"pilot", "production"} and not any(auth.values()):
        issues.append("pilot and production modes require an API key or configured durable user.")
    if RUNTIME_MODE == "production" and not (auth["admin_key"] or auth["admin_user"]):
        issues.append("production mode requires an administrator API key or administrator user.")
    if RUNTIME_MODE in {"pilot", "production"} and DEMO_MODE:
        issues.append("demo mode must be disabled in pilot and production modes.")
    if RUNTIME_MODE in {"pilot", "production"} and not os.getenv("OPTIVOX_BIOMETRIC_KEY", "").strip():
        issues.append("pilot and production modes require OPTIVOX_BIOMETRIC_KEY for encrypted local embeddings.")
    if RUNTIME_MODE in {"pilot", "production"} and not EVIDENCE_ENCRYPTION_ENABLED:
        issues.append("pilot and production modes require OPTIVOX_EVIDENCE_ENCRYPTION_ENABLED=true.")
    if EVIDENCE_ENCRYPTION_ENABLED or RUNTIME_MODE in {"pilot", "production"}:
        if not EVIDENCE_ENCRYPTION_KEY:
            issues.append("encrypted evidence requires OPTIVOX_EVIDENCE_ENCRYPTION_KEY.")
        elif len(EVIDENCE_ENCRYPTION_KEY) < 32:
            issues.append("OPTIVOX_EVIDENCE_ENCRYPTION_KEY must contain at least 32 characters.")
        if not EVIDENCE_ENCRYPTION_KEY_ID or len(EVIDENCE_ENCRYPTION_KEY_ID) > 64:
            issues.append("OPTIVOX_EVIDENCE_ENCRYPTION_KEY_ID must be 1 to 64 characters.")
        if EVIDENCE_ENCRYPTION_KEYS_JSON:
            try:
                evidence_key_ring = json.loads(EVIDENCE_ENCRYPTION_KEYS_JSON)
            except (TypeError, ValueError, json.JSONDecodeError):
                evidence_key_ring = None
                issues.append("OPTIVOX_EVIDENCE_ENCRYPTION_KEYS_JSON must be valid JSON.")
            if evidence_key_ring is not None:
                if not isinstance(evidence_key_ring, dict) or len(evidence_key_ring) > 4:
                    issues.append("OPTIVOX_EVIDENCE_ENCRYPTION_KEYS_JSON must contain at most four keys.")
                elif any(not str(key).strip() or len(str(value or "").strip()) < 32 for key, value in evidence_key_ring.items()):
                    issues.append("every evidence key-ring entry must have an ID and a 32-character-or-longer key.")
                elif EVIDENCE_ENCRYPTION_KEY_ID not in evidence_key_ring:
                    issues.append("OPTIVOX_EVIDENCE_ENCRYPTION_KEY_ID must exist in the configured evidence key ring.")
    if RUNTIME_MODE in {"pilot", "production"} and len(BACKUP_SIGNING_KEY) < 32:
        issues.append("pilot and production modes require OPTIVOX_BACKUP_SIGNING_KEY with at least 32 characters.")
    if RUNTIME_MODE in {"pilot", "production"} and not BACKUP_ENCRYPTION_ENABLED:
        issues.append("pilot and production modes require OPTIVOX_BACKUP_ENCRYPTION_ENABLED=true.")
    if BACKUP_ENCRYPTION_ENABLED or RUNTIME_MODE in {"pilot", "production"}:
        if not BACKUP_ENCRYPTION_KEY:
            issues.append("encrypted backups require OPTIVOX_BACKUP_ENCRYPTION_KEY.")
        elif len(BACKUP_ENCRYPTION_KEY) < 32:
            issues.append("OPTIVOX_BACKUP_ENCRYPTION_KEY must contain at least 32 characters.")
        if not BACKUP_ENCRYPTION_KEY_ID or len(BACKUP_ENCRYPTION_KEY_ID) > 64:
            issues.append("OPTIVOX_BACKUP_ENCRYPTION_KEY_ID must be 1 to 64 characters.")
        if BACKUP_ENCRYPTION_KEYS_JSON:
            try:
                key_ring = json.loads(BACKUP_ENCRYPTION_KEYS_JSON)
            except (TypeError, ValueError, json.JSONDecodeError):
                key_ring = None
                issues.append("OPTIVOX_BACKUP_ENCRYPTION_KEYS_JSON must be valid JSON.")
            if key_ring is not None:
                if not isinstance(key_ring, dict) or len(key_ring) > 4:
                    issues.append("OPTIVOX_BACKUP_ENCRYPTION_KEYS_JSON must contain at most four keys.")
                elif any(not str(key).strip() or len(str(value or "").strip()) < 32 for key, value in key_ring.items()):
                    issues.append("every backup key-ring entry must have an ID and a 32-character-or-longer key.")
                elif BACKUP_ENCRYPTION_KEY_ID not in key_ring:
                    issues.append("OPTIVOX_BACKUP_ENCRYPTION_KEY_ID must exist in the configured backup key ring.")
    if RUNTIME_MODE in {"pilot", "production"} and len(AUDIT_SIGNING_KEY) < 32:
        issues.append("pilot and production modes require OPTIVOX_AUDIT_SIGNING_KEY with at least 32 characters.")
    if BACKUP_SCHEDULER_ENABLED and BACKUP_INTERVAL_MINUTES < 5:
        issues.append("backup scheduling requires OPTIVOX_BACKUP_INTERVAL_MINUTES of at least 5.")
    if BACKUP_MAX_AGE_MINUTES < BACKUP_INTERVAL_MINUTES:
        issues.append("OPTIVOX_BACKUP_MAX_AGE_MINUTES must be at least the backup interval.")
    if RUNTIME_MODE in {"pilot", "production"} and not (BACKUP_SCHEDULER_ENABLED or EXTERNAL_BACKUP_CONFIRMED):
        issues.append(
            "pilot and production modes require the OptiVox backup scheduler or "
            "OPTIVOX_EXTERNAL_BACKUP_CONFIRMED=true."
        )
    if BACKUP_RETENTION_COUNT < 2:
        issues.append("OPTIVOX_BACKUP_RETENTION_COUNT must be at least 2.")
    if BACKUP_RETENTION_DAYS < 1:
        issues.append("OPTIVOX_BACKUP_RETENTION_DAYS must be positive.")
    if SYNC_ENABLED:
        if not SYNC_ENDPOINT:
            issues.append("enabled synchronization requires OPTIVOX_SYNC_ENDPOINT.")
        if len(SYNC_SECRET) < 32:
            issues.append("enabled synchronization requires OPTIVOX_SYNC_SECRET with at least 32 characters.")
        if not SYNC_ALLOWED_HOSTS:
            issues.append("enabled synchronization requires OPTIVOX_SYNC_ALLOWED_HOSTS.")
        if SYNC_ENDPOINT:
            try:
                from deployment_security import validate_control_plane_url
                validate_control_plane_url(SYNC_ENDPOINT, RUNTIME_MODE, SYNC_ALLOWED_HOSTS)
            except Exception as exc:
                issues.append(f"invalid synchronization endpoint: {exc}")
        algorithm = "ed25519" if SYNC_PRIVATE_KEY_PATH else "hmac-sha256"
        if algorithm == "ed25519" and not SYNC_KEY_ID:
            issues.append("Ed25519 synchronization requires OPTIVOX_SYNC_KEY_ID.")
        if SYNC_PRIVATE_KEY_PATH:
            key_path = Path(SYNC_PRIVATE_KEY_PATH).expanduser()
            if not key_path.is_absolute():
                key_path = (PROJECT_ROOT / key_path).resolve()
            if not key_path.is_file():
                issues.append("OPTIVOX_SYNC_PRIVATE_KEY_PATH must point to an existing private key file.")
    issues.extend(sync_device_registry_issues())

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
BACKUP_STATUS_PATH = runtime_path("backup_status.json")
LIVENESS_METRICS_PATH = runtime_path("liveness_metrics.json")
HEALTH_EVENTS_PATH = runtime_path("health_events.jsonl")
SECURITY_ZONES_PATH = runtime_path("security_zones.json")
