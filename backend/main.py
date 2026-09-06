from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGINS
from .platform_schema import ensure_platform_schema
from .routes import academic, analytics, attendance, auth, commands, events, health, incidents, live, operations, people, system
from .services.auth_service import bootstrap_configured_users

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("optivox.backend")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_platform_schema()
    bootstrap_configured_users()
    logger.info("OptiVox backend started")
    yield
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
    allow_headers=["*"],
)

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
]:
    app.include_router(router)
