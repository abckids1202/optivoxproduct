from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from ..security import require_permission
from pydantic import BaseModel, Field

from ..services.sync_service import SyncIngestError, ingest_batch, set_device_status, sync_status


router = APIRouter(prefix="/api/sync", tags=["sync"])


class DeviceStatusRequest(BaseModel):
    status: str = Field(pattern="^(active|revoked|disabled)$")


@router.get("/status")
def status(actor: str = Depends(require_permission("system.view"))):
    return sync_status()


@router.post("/devices/{device_id}/status")
def device_status(
    device_id: str,
    payload: DeviceStatusRequest,
    actor: str = Depends(require_permission("system.manage")),
):
    return set_device_status(device_id, payload.status, actor_id=actor)


@router.post("/ingest")
async def ingest(
    request: Request,
    x_optivox_signature: str | None = Header(default=None),
    x_optivox_signature_algorithm: str | None = Header(default=None),
    x_optivox_key_id: str | None = Header(default=None),
    x_optivox_device: str | None = Header(default=None),
    x_optivox_site: str | None = Header(default=None),
    x_optivox_organization: str | None = Header(default=None),
):
    """Receive one authenticated, contiguous edge batch."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_SYNC_JSON", "message": "Sync body must be valid JSON."}) from exc
    try:
        return ingest_batch(
            body,
            x_optivox_signature or "",
            header_scope={
                "device": x_optivox_device or "",
                "site": x_optivox_site or "",
                "organization": x_optivox_organization or "",
            },
            signature_algorithm=x_optivox_signature_algorithm or "",
            key_id=x_optivox_key_id or "",
        )
    except SyncIngestError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": "SYNC_REJECTED", "message": str(exc), **exc.details}) from exc
