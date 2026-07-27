from __future__ import annotations

from fastapi import APIRouter

from ..services import attendance_service as svc

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


@router.get("/today")
def today():
    return svc.today_attendance()


@router.get("")
def list_attendance(limit: int = 200, offset: int = 0, person_id: int | None = None):
    return svc.list_attendance(limit=limit, offset=offset, person_id=person_id)


@router.get("/summary")
def summary():
    return svc.attendance_summary()


@router.get("/person/{person_id}")
def person(person_id: int):
    return svc.person_attendance(person_id)


@router.get("/export")
def export():
    return svc.export_csv()


@router.post("/{person_id}/clock-in")
def clock_in(person_id: int):
    return svc.clock_in(person_id)


@router.post("/{person_id}/clock-out")
def clock_out(person_id: int):
    return svc.clock_out(person_id)
