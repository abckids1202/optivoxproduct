from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..services import attendance_service as svc
from ..security import require_permission

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


class CorrectionRequest(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    clock_in: str | None = Field(default=None, max_length=64)
    clock_out: str | None = Field(default=None, max_length=64)
    late_minutes: int = Field(default=0, ge=0, le=1440)
    reason: str = Field(min_length=3, max_length=500)


class AbsenceRequest(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    status: str = Field(min_length=1, max_length=20)
    subject: str | None = Field(default=None, max_length=120)
    reason: str | None = Field(default=None, max_length=500)


@router.get("/today")
def today(actor: str = Depends(require_permission("attendance.view"))):
    return svc.today_attendance()


@router.get("")
def list_attendance(limit: int = 200, offset: int = 0, person_id: int | None = None, actor: str = Depends(require_permission("attendance.view"))):
    return svc.list_attendance(limit=limit, offset=offset, person_id=person_id)


@router.get("/summary")
def summary(actor: str = Depends(require_permission("attendance.view"))):
    return svc.attendance_summary()


@router.get("/calendar")
def calendar(year: int | None = None, month: int | None = None, actor: str = Depends(require_permission("attendance.view"))):
    return svc.attendance_calendar(year=year, month=month)


@router.get("/person/{person_id}")
def person(person_id: int, actor: str = Depends(require_permission("attendance.view"))):
    return svc.person_attendance(person_id)


@router.get("/absences")
def absences(person_id: int | None = None, start: str | None = None, end: str | None = None, actor: str = Depends(require_permission("attendance.view"))):
    return svc.list_absences(person_id=person_id, start=start, end=end)


@router.get("/export")
def export(actor: str = Depends(require_permission("attendance.export"))):
    return svc.export_csv()


@router.post("/{person_id}/clock-in")
def clock_in(person_id: int, actor: str = Depends(require_permission("attendance.manual"))):
    return svc.clock_in(person_id, actor_id=actor)


@router.post("/{person_id}/clock-out")
def clock_out(person_id: int, actor: str = Depends(require_permission("attendance.manual"))):
    return svc.clock_out(person_id, actor_id=actor)


@router.post("/{person_id}/correct")
def correct(person_id: int, payload: CorrectionRequest, actor: str = Depends(require_permission("attendance.manage"))):
    return svc.correct_attendance(person_id, payload.date, payload.clock_in, payload.clock_out, payload.late_minutes, payload.reason, actor_id=actor)


@router.post("/{person_id}/absence")
def absence(person_id: int, payload: AbsenceRequest, actor: str = Depends(require_permission("attendance.manage"))):
    return svc.record_absence(person_id, payload.date, payload.status, payload.subject, payload.reason, actor_id=actor)
