from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from ..services import command_service as svc
from ..security import require_permission

router = APIRouter(prefix="/api", tags=["commands"])


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)


@router.post("/commands")
def command(req: CommandRequest, request: Request,
            actor: str = Depends(require_permission("commands.execute")),
            x_optivox_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    permission = svc.COMMAND_PERMISSIONS.get(req.command)
    if permission:
        actor = require_permission(permission)(request, x_optivox_key, authorization)
    payload = {**req.payload, "actor_id": actor}
    return svc.create_command(req.command, payload, idempotency_key=idempotency_key)


@router.get("/commands/{command_id}")
def status(command_id: str, actor: str = Depends(require_permission("operations.view"))):
    return svc.command_status(command_id)


@router.post("/enrollment/start", dependencies=[Depends(require_permission("biometric.enroll"))])
def enrollment_start(payload: dict, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("start_enrollment", payload, idempotency_key=idempotency_key)


@router.post("/enrollment/register-visible", dependencies=[Depends(require_permission("biometric.enroll"))])
def register_visible(payload: dict, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("register_visible_unknown", payload, idempotency_key=idempotency_key)


@router.post("/enrollment/cancel", dependencies=[Depends(require_permission("biometric.enroll"))])
def cancel_enrollment():
    return svc.create_command("cancel_enrollment", {})


@router.post("/enrollment/confirm")
def confirm_enrollment(payload: dict, actor: str = Depends(require_permission("biometric.enroll")), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("confirm_enrollment", {**payload, "actor_id": actor}, idempotency_key=idempotency_key)


@router.post("/people/{person_name}/retrain")
def retrain_person(person_name: str, actor: str = Depends(require_permission("biometric.manage")), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("retrain_person", {"name": person_name, "actor_id": actor}, idempotency_key=idempotency_key)


@router.post("/people/{person_name}/disable")
def disable_person(person_name: str, actor: str = Depends(require_permission("biometric.manage")), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("disable_person", {"name": person_name, "actor_id": actor}, idempotency_key=idempotency_key)


@router.post("/people/merge")
def merge_people(payload: dict, actor: str = Depends(require_permission("biometric.merge")), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("merge_people", {**payload, "actor_id": actor}, idempotency_key=idempotency_key)


@router.delete("/people/{person_name}")
def delete_person(person_name: str, confirm: bool = False, actor: str = Depends(require_permission("biometric.delete")), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return svc.create_command("delete_person", {"name": person_name, "confirm": confirm, "actor_id": actor}, idempotency_key=idempotency_key)


@router.get("/enrollment/status")
def enrollment_status(actor: str = Depends(require_permission("biometric.enroll"))):
    return svc.enrollment_status()
