from fastapi import APIRouter, Depends

from ..services import analytics_service as svc
from ..security import require_operator

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
def overview(actor: str = Depends(require_operator)):
    return svc.overview()


@router.get("/attendance")
def attendance(days: int = 7, actor: str = Depends(require_operator)):
    return svc.attendance(days)


@router.get("/security")
def security(days: int = 30, actor: str = Depends(require_operator)):
    return svc.security(days)


@router.get("/objects")
def objects(actor: str = Depends(require_operator)):
    return svc.objects()
