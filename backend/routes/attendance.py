from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..services import attendance_service as svc
from ..security import require_admin, require_operator

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


class CorrectionRequest(BaseModel):
    date: str = Field(min_length=10, max_length=10)
    clock_in: str | None = Field(default=None, max_length=64)
    clock_out: str | None = Field(default=None, max_length=64)
    late_minutes: int = Field(default=0, ge=0, le=1440)
    reason: str = Field(min_length=3, max_length=500)


@router.get("/today")
def today():
    return svc.today_attendance()


@router.get("")
def list_attendance(limit: int = 200, offset: int = 0, person_id: int | None = None):
    return svc.list_attendance(limit=limit, offset=offset, person_id=person_id)


@router.get("/summary")
def summary():
    return svc.attendance_summary()


@router.get("/calendar")
def calendar(year: int | None = None, month: int | None = None):
    return svc.attendance_calendar(year=year, month=month)


@router.get("/person/{person_id}")
def person(person_id: int):
    return svc.person_attendance(person_id)


@router.get("/export")
def export():
    return svc.export_csv()


@router.post("/{person_id}/clock-in", dependencies=[Depends(require_operator)])
def clock_in(person_id: int):
    return svc.clock_in(person_id)


@router.post("/{person_id}/clock-out", dependencies=[Depends(require_operator)])
def clock_out(person_id: int):
    return svc.clock_out(person_id)


@router.post("/{person_id}/correct", dependencies=[Depends(require_admin)])
def correct(person_id: int, payload: CorrectionRequest):
    return svc.correct_attendance(person_id, payload.date, payload.clock_in, payload.clock_out, payload.late_minutes, payload.reason)
