from __future__ import annotations

from fastapi import APIRouter

from ..services import academic_service as svc

router = APIRouter(prefix="/api/academic", tags=["academic"])


@router.get("/overview")
def overview(year: int | None = None, month: int | None = None):
    return svc.overview(year=year, month=month)
