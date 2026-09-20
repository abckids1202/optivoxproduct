from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services import command_service
from ..services import system_service as svc
from ..services.audit_service import integrity_report
from ..services.audit_service import record_action
from ..services.cybersecurity_service import record_cyber_event
from ..services.storage_service import backup_database, purge_backups, purge_expired_evidence, rehearse_latest_backup
from ..services.retention_service import DEFAULT_RETENTION_DAYS, MAX_RETENTION_DAYS, preview as retention_preview, purge as purge_retention
from ..services import outbox_service
from ..database import database_maintenance, fetch_all
from ..config import EVIDENCE_RETENTION_DAYS, ORGANIZATION_ID, SITE_ID
from ..security import require_permission

router = APIRouter(prefix="/api/system", tags=["system"])


def _redact_policy_value(value):
    """Return policy data without exposing credentials or private endpoints."""
    sensitive_fragments = (
        "secret", "token", "password", "api_key", "apikey",
        "private_key", "webhook_url", "smtp_pass", "bot_token",
    )
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(fragment in normalized for fragment in sensitive_fragments):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact_policy_value(item)
        return result
    if isinstance(value, list):
        return [_redact_policy_value(item) for item in value]
    return value


class VehicleCalibrationRequest(BaseModel):
    homography: list[list[float]] = Field(min_length=3, max_length=3)
    image_width: int = Field(gt=0, le=10000)
    image_height: int = Field(gt=0, le=10000)
    version: str = Field(default="1", max_length=64)


class SecurityZonesRequest(BaseModel):
    zones: list[dict] = Field(default_factory=list, max_length=64)


class OutboxRequeueRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)


class OutboxPurgeRequest(BaseModel):
    retention_days: int = Field(default=30, ge=1, le=3650)
    execute: bool = False


class DatabaseMaintenanceRequest(BaseModel):
    full_integrity: bool = False
    checkpoint: bool = False
    optimize: bool = True
    repair_schema: bool = False


class DatabaseRetentionRequest(BaseModel):
    retention_days: int = Field(default=DEFAULT_RETENTION_DAYS, ge=1, le=MAX_RETENTION_DAYS)
    execute: bool = False


class BackupRetentionRequest(BaseModel):
    retention_days: int = Field(default=30, ge=1, le=3650)
    retention_count: int = Field(default=24, ge=2, le=500)
    execute: bool = False


@router.get("/status")
def status(actor: str = Depends(require_permission("system.view"))):
    return svc.system_status()


@router.get("/policy")
def policy_history(actor: str = Depends(require_permission("system.view"))):
    """Return the scoped policy history without exposing sensitive config."""
    rows = fetch_all(
        """select policy_version, declared_version, organization_id, site_id,
                  device_id, document_json, issues_json, valid,
                  first_seen_at, last_seen_at
             from policy_snapshots
            where organization_id=? and site_id=?
            order by last_seen_at desc, policy_version desc
            limit 100""",
        [ORGANIZATION_ID, SITE_ID],
    )
    history = []
    for row in rows:
        try:
            document = json.loads(row.get("document_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            document = {"status": "CORRUPT_POLICY_DOCUMENT"}
        try:
            issues = json.loads(row.get("issues_json") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            issues = ["CORRUPT_POLICY_ISSUES"]
        history.append({
            "policy_version": row.get("policy_version"),
            "declared_version": row.get("declared_version"),
            "organization_id": row.get("organization_id"),
            "site_id": row.get("site_id"),
            "device_id": row.get("device_id"),
            "valid": bool(row.get("valid")),
            "issues": issues if isinstance(issues, list) else [str(issues)],
            "document": _redact_policy_value(document),
            "first_seen_at": row.get("first_seen_at"),
            "last_seen_at": row.get("last_seen_at"),
        })
    return {"current": history[0] if history else None, "history": history}


@router.get("/models")
def models(actor: str = Depends(require_permission("system.view"))):
    return svc.models_status()


@router.get("/database")
def database(actor: str = Depends(require_permission("system.view"))):
    return svc.database_status()


@router.post("/database/maintenance")
def database_maintenance_route(
    payload: DatabaseMaintenanceRequest,
    request: Request,
    actor: str = Depends(require_permission("system.manage")),
):
    """Run explicit maintenance; ordinary health requests stay lightweight."""
    result = database_maintenance(
        full_integrity=payload.full_integrity,
        checkpoint=payload.checkpoint,
        optimize=payload.optimize,
        repair_schema=payload.repair_schema,
    )
    details = {
        "full_integrity": payload.full_integrity,
        "checkpoint": payload.checkpoint,
        "optimize": payload.optimize,
        "repair_schema": payload.repair_schema,
        "integrity_check": result.get("integrity_check"),
        "wal_checkpoint": result.get("wal_checkpoint"),
    }
    record_cyber_event(
        "CONFIGURATION_CHANGE",
        source="system_api",
        actor_id=actor,
        actor_type="user",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"operation": "database_maintenance", **details},
    )
    record_action("database.maintenance", "database", None, details, actor_type="operator", actor_id=actor)
    result.pop("path", None)
    return result


@router.post("/database/retention")
def database_retention(
    payload: DatabaseRetentionRequest,
    request: Request,
    actor: str = Depends(require_permission("system.manage")),
):
    """Preview or explicitly purge unlinked low-level telemetry."""
    result = purge_retention(payload.retention_days, actor_id=actor) if payload.execute else retention_preview(payload.retention_days)
    if payload.execute:
        record_cyber_event(
            "CONFIGURATION_CHANGE", source="system_api", actor_id=actor, actor_type="user",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"operation": "database_retention_purge", "retention_days": payload.retention_days,
                     "deleted_total": result.get("deleted_total", 0)},
        )
    return result


@router.get("/storage")
def storage(actor: str = Depends(require_permission("system.view"))):
    return svc.storage_status()


@router.get("/alerts")
def alerts(actor: str = Depends(require_permission("system.view"))):
    return svc.alerts_status()


@router.get("/performance")
def performance(actor: str = Depends(require_permission("system.view"))):
    return svc.performance_report()


@router.get("/security-zones")
def security_zones(actor: str = Depends(require_permission("system.view"))):
    return svc.security_zones_status()


@router.get("/outbox")
def outbox(actor: str = Depends(require_permission("system.view"))):
    """Expose queue health without returning operational payloads."""
    return outbox_service.summary()


@router.post("/outbox/requeue")
def requeue_outbox(payload: OutboxRequeueRequest, actor: str = Depends(require_permission("system.manage"))):
    return {"ok": outbox_service.requeue_dead_letter(payload.event_id, actor_id=actor), "event_id": payload.event_id}


@router.post("/outbox/purge")
def purge_outbox(payload: OutboxPurgeRequest, actor: str = Depends(require_permission("system.manage"))):
    return outbox_service.purge_sent(
        payload.retention_days,
        execute_delete=payload.execute,
        actor_id=actor,
    )


@router.post("/vehicle-calibration")
def vehicle_calibration(payload: VehicleCalibrationRequest, request: Request,
                        actor: str = Depends(require_permission("system.manage"))):
    """Queue a validated image-to-ground calibration for the local edge runtime."""
    return command_service.create_command(
        "set_vehicle_calibration",
        {"calibration": payload.dict(), "actor_id": actor},
    )


@router.post("/security-zones")
def update_security_zones(payload: SecurityZonesRequest, actor: str = Depends(require_permission("system.manage"))):
    return command_service.create_command(
        "set_security_zones",
        {"zones": payload.zones, "actor_id": actor},
    )


@router.get("/audit-integrity")
def audit_integrity(request: Request, actor: str = Depends(require_permission("system.view"))):
    result = integrity_report()
    if not result.get("ok"):
        record_cyber_event(
            "DATABASE_INTEGRITY_FAILURE", source="audit_integrity", actor_id=actor,
            actor_type="user", ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"), details={"report": result},
            dedupe_key=f"audit-integrity:{result.get('record_id')}:{result.get('reason')}",
        )
    return result


@router.post("/backup")
def backup(request: Request, actor: str = Depends(require_permission("system.manage"))):
    result = backup_database()
    record_cyber_event("CONFIGURATION_CHANGE", source="system_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details={"operation": "database_backup", "sha256": result["sha256"]})
    record_action("storage.backup", "database", None, {"sha256": result["sha256"]}, actor_type="operator", actor_id=actor)
    return {"ok": True, "filename": result["path"].split("\\")[-1], "sha256": result["sha256"]}


@router.post("/backups/retention")
def backup_retention(
    payload: BackupRetentionRequest,
    request: Request,
    actor: str = Depends(require_permission("system.manage")),
):
    """Preview or explicitly delete only verified, aged backup pairs."""
    result = purge_backups(
        payload.retention_days,
        payload.retention_count,
        execute_delete=payload.execute,
    )
    if payload.execute:
        details = {
            "operation": "backup_retention_purge",
            "retention_days": payload.retention_days,
            "retention_count": payload.retention_count,
            "deleted_count": result.get("deleted_count", 0),
            "failure_count": len(result.get("failures", [])),
        }
        record_cyber_event(
            "CONFIGURATION_CHANGE", source="system_api", actor_id=actor, actor_type="user",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"), details=details,
        )
        record_action("storage.backup_retention_purge", "backup", None, details, actor_type="operator", actor_id=actor)
    return result


@router.post("/backups/recovery-drill")
def backup_recovery_drill(request: Request, actor: str = Depends(require_permission("system.manage"))):
    """Prove the newest verified backup can be restored without touching live data."""
    result = rehearse_latest_backup()
    details = {
        "operation": "backup_recovery_drill",
        "status": result.get("status"),
        "backup_name": result.get("backup_name"),
        "reason": result.get("reason"),
        "integrity_check": result.get("integrity_check"),
        "foreign_key_violations": result.get("foreign_key_violations"),
    }
    record_cyber_event(
        "DATABASE_INTEGRITY_FAILURE" if result.get("status") != "success" else "CONFIGURATION_CHANGE",
        source="system_api",
        actor_id=actor,
        actor_type="user",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details=details,
    )
    record_action(
        "storage.backup_recovery_drill",
        "backup",
        result.get("backup_name"),
        details,
        actor_type="operator",
        actor_id=actor,
    )
    return result


@router.post("/purge-evidence")
def purge_evidence(request: Request, actor: str = Depends(require_permission("system.manage"))):
    referenced = fetch_all(
        "select path from incident_evidence where organization_id=? and site_id=? and path is not null",
        [ORGANIZATION_ID, SITE_ID],
    )
    result = purge_expired_evidence(
        EVIDENCE_RETENTION_DAYS,
        protected_paths=[row["path"] for row in referenced],
    )
    details = {"operation": "evidence_retention_purge", **result, "retention_days": EVIDENCE_RETENTION_DAYS}
    record_cyber_event("CONFIGURATION_CHANGE", source="system_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details=details)
    record_action("evidence.retention_purge", "evidence", None, details, actor_type="operator", actor_id=actor)
    return {"ok": True, **result, "retention_days": EVIDENCE_RETENTION_DAYS}


@router.post("/test-alert", dependencies=[Depends(require_permission("system.manage"))])
def test_alert():
    return command_service.create_command("test_alert", {})


@router.post("/save-snapshot", dependencies=[Depends(require_permission("live.view"))])
def save_snapshot():
    return command_service.create_command("save_snapshot", {})


@router.post("/demo-reset")
def demo_reset(request: Request, payload: dict | None = None, actor: str = Depends(require_permission("system.manage"))):
    record_cyber_event("CONFIGURATION_CHANGE", source="system_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details={"operation": "demo_reset"})
    return command_service.create_command("reset_demo_data", payload or {})
