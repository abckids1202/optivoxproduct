from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import BACKUP_INTERVAL_MINUTES, BACKUP_RETENTION_COUNT, BACKUP_RETENTION_DAYS, BACKUP_SCHEDULER_ENABLED, FRONTEND_ORIGINS, MAX_REQUEST_BODY_BYTES, RUNTIME_MODE, validate_runtime_configuration
from .platform_schema import ensure_platform_schema
from .routes import academic, analytics, attendance, auth, commands, cybersecurity, events, health, incidents, live, operations, people, system, sync
from .services.auth_service import bootstrap_configured_users
from .services.backup_scheduler import BackupScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("optivox.backend")

# Validate before the import-time schema compatibility check can touch the
# operational database. Lifespan repeats this check for supervised startups.
validate_runtime_configuration()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    validate_runtime_configuration()
    ensure_platform_schema()
    bootstrap_configured_users()
    backup_scheduler = None
    if BACKUP_SCHEDULER_ENABLED:
        backup_scheduler = BackupScheduler(
            interval_minutes=BACKUP_INTERVAL_MINUTES,
            retention_days=BACKUP_RETENTION_DAYS,
            retention_count=BACKUP_RETENTION_COUNT,
        )
        backup_scheduler.start()
        logger.info("OptiVox backup scheduler started (%s minutes)", BACKUP_INTERVAL_MINUTES)
    logger.info("OptiVox backend started in %s mode", RUNTIME_MODE)
    try:
        yield
    finally:
        if backup_scheduler is not None:
            backup_scheduler.stop()
            backup_scheduler.join(timeout=5)
        logger.info("OptiVox backend stopped")


app = FastAPI(
    title="OptiVox Backend",
    description="Local FastAPI bridge for OptiVox computer vision, attendance, and security data.",
    version="1.0.0-exhibition",
    lifespan=lifespan,
)

# Keep direct imports and lightweight ASGI clients consistent with production
# startup. This is idempotent and does not touch biometric data.
ensure_platform_schema()

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization", "Content-Type", "X-Optivox-Key", "X-CSRF-Token", "Idempotency-Key",
        "X-Optivox-Signature", "X-Optivox-Signature-Algorithm", "X-Optivox-Key-Id",
        "X-Optivox-Device", "X-Optivox-Site", "X-Optivox-Organization",
    ],
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=()")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    if RUNTIME_MODE == "production":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    """Reject oversized API requests before parsing or persisting their body."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": {"code": "INVALID_CONTENT_LENGTH", "message": "Content-Length must be a valid integer."}},
            )
        if declared < 0 or declared > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": {"code": "REQUEST_BODY_TOO_LARGE", "message": "Request body exceeds the configured limit."}},
            )
    return await call_next(request)

for router in [
    health.router,
    auth.router,
    live.router,
    attendance.router,
    events.router,
    incidents.router,
    operations.router,
    people.router,
    analytics.router,
    commands.router,
    system.router,
    academic.router,
    cybersecurity.router,
    sync.router,
]:
    app.include_router(router)
