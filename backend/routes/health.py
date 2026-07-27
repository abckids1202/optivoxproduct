from datetime import datetime

from fastapi import APIRouter

from ..config import APP_VERSION, DATABASE_PATH, RUNTIME_DIR, TIMEZONE
from ..database import get_connection
from ..services.runtime_service import heartbeat_state

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    database = "missing"
    if DATABASE_PATH.exists():
        try:
            with get_connection() as con:
                con.execute("select 1").fetchone()
            database = "connected"
        except Exception:
            database = "error"
    hb = heartbeat_state()
    status = "ok" if database == "connected" else "degraded"
    return {
        "status": status,
        "backend": "online",
        "database": database,
        "engine": hb["status"],
        "runtime": "available" if RUNTIME_DIR.exists() else "missing",
        "version": APP_VERSION,
        "timestamp": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
    }

