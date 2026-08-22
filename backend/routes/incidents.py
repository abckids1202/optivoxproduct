from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..security import require_operator
from ..services import incident_service as svc

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


class ReviewRequest(BaseModel):
    action: str = Field(min_length=1, max_length=20)
    note: str | None = Field(default=None, max_length=500)


@router.get("")
def incidents(limit: int = 100, status: str | None = None):
    return svc.list_incidents(limit=limit, status=status)


@router.get("/{incident_id}")
def incident(incident_id: int):
    return svc.get_incident(incident_id)


@router.post("/{incident_id}/review", dependencies=[Depends(require_operator)])
def review(incident_id: int, payload: ReviewRequest):
    return svc.review_incident(incident_id, payload.action, payload.note)
