from fastapi import APIRouter

from ..services import analytics_service as svc

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
def overview():
    return svc.overview()


@router.get("/attendance")
def attendance(days: int = 7):
    return svc.attendance(days)


@router.get("/security")
def security():
    return svc.security()


@router.get("/objects")
def objects():
    return svc.objects()

