from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGINS
from .routes import academic, analytics, attendance, commands, events, health, live, people, system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("optivox.backend")

app = FastAPI(
    title="OptiVox Backend",
    description="Local FastAPI bridge for OptiVox computer vision, attendance, and security data.",
    version="1.0.0-exhibition",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

for router in [
    health.router,
    live.router,
    attendance.router,
    events.router,
    people.router,
    analytics.router,
    commands.router,
    system.router,
    academic.router,
]:
    app.include_router(router)


@app.on_event("startup")
async def startup():
    logger.info("OptiVox backend started")


@app.on_event("shutdown")
async def shutdown():
    logger.info("OptiVox backend stopped")
