from __future__ import annotations

from fastapi import APIRouter, Depends

from ..services import command_service
from ..services import system_service as svc
from ..security import require_operator

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
def status(actor: str = Depends(require_operator)):
    return svc.system_status()


@router.get("/models")
def models(actor: str = Depends(require_operator)):
    return svc.models_status()


@router.get("/database")
def database(actor: str = Depends(require_operator)):
    return svc.database_status()


@router.get("/storage")
def storage(actor: str = Depends(require_operator)):
    return svc.storage_status()


@router.get("/alerts")
def alerts(actor: str = Depends(require_operator)):
    return svc.alerts_status()


@router.get("/performance")
def performance(actor: str = Depends(require_operator)):
    return svc.performance_report()


@router.post("/test-alert", dependencies=[Depends(require_operator)])
def test_alert():
    return command_service.create_command("test_alert", {})


@router.post("/save-snapshot", dependencies=[Depends(require_operator)])
def save_snapshot():
    return command_service.create_command("save_snapshot", {})


@router.post("/demo-reset", dependencies=[Depends(require_operator)])
def demo_reset(payload: dict | None = None):
    return command_service.create_command("reset_demo_data", payload or {})
