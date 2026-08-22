from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from ..services import command_service as svc
from ..security import require_operator

router = APIRouter(prefix="/api", tags=["commands"])


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)


@router.post("/commands", dependencies=[Depends(require_operator)])
def command(req: CommandRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command(req.command, req.payload, idempotency_key=idempotency_key)


@router.get("/commands/{command_id}")
def status(command_id: str):
    return svc.command_status(command_id)


@router.post("/enrollment/start", dependencies=[Depends(require_operator)])
def enrollment_start(payload: dict, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("start_enrollment", payload, idempotency_key=idempotency_key)


@router.post("/enrollment/register-visible", dependencies=[Depends(require_operator)])
def register_visible(payload: dict, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("register_visible_unknown", payload, idempotency_key=idempotency_key)


@router.post("/enrollment/cancel", dependencies=[Depends(require_operator)])
def cancel_enrollment():
    return svc.create_command("cancel_enrollment", {})


@router.get("/enrollment/status")
def enrollment_status():
    return svc.enrollment_status()
