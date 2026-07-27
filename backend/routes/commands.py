from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import command_service as svc

router = APIRouter(prefix="/api", tags=["commands"])


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)


@router.post("/commands")
def command(req: CommandRequest):
    return svc.create_command(req.command, req.payload)


@router.get("/commands/{command_id}")
def status(command_id: str):
    return svc.command_status(command_id)


@router.post("/enrollment/start")
def enrollment_start(payload: dict):
    return svc.create_command("start_enrollment", payload)


@router.post("/enrollment/register-visible")
def register_visible(payload: dict):
    return svc.create_command("register_visible_unknown", payload)


@router.post("/enrollment/cancel")
def cancel_enrollment():
    return svc.create_command("cancel_enrollment", {})


@router.get("/enrollment/status")
def enrollment_status():
    return svc.enrollment_status()

