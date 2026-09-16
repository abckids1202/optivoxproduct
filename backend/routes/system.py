from __future__ import annotations

from fastapi import APIRouter, Depends

from ..services import command_service
from ..services import system_service as svc
from ..services.audit_service import integrity_report
from ..services.audit_service import record_action
from ..services.storage_service import backup_database, purge_expired_evidence
from ..config import EVIDENCE_RETENTION_DAYS
from ..security import require_permission

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
def status(actor: str = Depends(require_permission("system.view"))):
    return svc.system_status()


@router.get("/models")
def models(actor: str = Depends(require_permission("system.view"))):
    return svc.models_status()


@router.get("/database")
def database(actor: str = Depends(require_permission("system.view"))):
    return svc.database_status()


@router.get("/storage")
def storage(actor: str = Depends(require_permission("system.view"))):
    return svc.storage_status()


@router.get("/alerts")
def alerts(actor: str = Depends(require_permission("system.view"))):
    return svc.alerts_status()


@router.get("/performance")
def performance(actor: str = Depends(require_permission("system.view"))):
    return svc.performance_report()


@router.get("/audit-integrity")
def audit_integrity(actor: str = Depends(require_permission("system.view"))):
    return integrity_report()


@router.post("/backup")
def backup(actor: str = Depends(require_permission("system.manage"))):
    result = backup_database()
    record_action("storage.backup", "database", None, {"sha256": result["sha256"]}, actor_type="operator", actor_id=actor)
    return {"ok": True, "filename": result["path"].split("\\")[-1], "sha256": result["sha256"]}


@router.post("/purge-evidence")
def purge_evidence(actor: str = Depends(require_permission("system.manage"))):
    deleted = purge_expired_evidence(EVIDENCE_RETENTION_DAYS)
    record_action("evidence.retention_purge", "evidence", None, {"deleted": deleted, "retention_days": EVIDENCE_RETENTION_DAYS}, actor_type="operator", actor_id=actor)
    return {"ok": True, "deleted": deleted, "retention_days": EVIDENCE_RETENTION_DAYS}


@router.post("/test-alert", dependencies=[Depends(require_permission("system.manage"))])
def test_alert():
    return command_service.create_command("test_alert", {})


@router.post("/save-snapshot", dependencies=[Depends(require_permission("live.view"))])
def save_snapshot():
    return command_service.create_command("save_snapshot", {})


@router.post("/demo-reset", dependencies=[Depends(require_permission("system.manage"))])
def demo_reset(payload: dict | None = None):
    return command_service.create_command("reset_demo_data", payload or {})
