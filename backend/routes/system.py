from __future__ import annotations

from fastapi import APIRouter, Depends

from ..services import command_service
from ..services import system_service as svc
from ..security import require_operator

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
def status():
    return svc.system_status()


@router.get("/models")
def models():
    return svc.models_status()


@router.get("/database")
def database():
    return svc.database_status()


@router.get("/storage")
def storage():
    return svc.storage_status()


@router.get("/alerts")
def alerts():
    return svc.alerts_status()


@router.post("/test-alert", dependencies=[Depends(require_operator)])
def test_alert():
    return command_service.create_command("test_alert", {})


@router.post("/save-snapshot", dependencies=[Depends(require_operator)])
def save_snapshot():
    return command_service.create_command("save_snapshot", {})


@router.post("/demo-reset", dependencies=[Depends(require_operator)])
def demo_reset(payload: dict | None = None):
    return command_service.create_command("reset_demo_data", payload or {})
