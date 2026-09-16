from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..services import academic_service as svc
from ..security import require_permission

router = APIRouter(prefix="/api/academic", tags=["academic"])


@router.get("/overview")
def overview(year: int | None = None, month: int | None = None, actor: str = Depends(require_permission("attendance.view"))):
    return svc.overview(year=year, month=month)


class ScheduleRequest(BaseModel):
    class_name: str = Field(min_length=1, max_length=120)
    subject: str | None = Field(default=None, max_length=120)
    weekday: int = Field(ge=0, le=6)
    start_time: str = Field(min_length=4, max_length=5)
    end_time: str | None = Field(default=None, max_length=5)
    grace_minutes: int = Field(default=0, ge=0, le=240)


@router.get("/schedules")
def schedules(active_only: bool = False, actor: str = Depends(require_permission("attendance.view"))):
    return svc.list_schedules(active_only=active_only)


@router.post("/schedules")
def add_schedule(payload: ScheduleRequest, actor: str = Depends(require_permission("attendance.manage"))):
    try:
        return svc.create_schedule(**payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_SCHEDULE", "message": str(exc)})


@router.delete("/schedules/{schedule_id}")
def deactivate_schedule(schedule_id: int, actor: str = Depends(require_permission("attendance.manage"))):
    try:
        return svc.deactivate_schedule(schedule_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"code": "SCHEDULE_NOT_FOUND", "message": str(exc)})
