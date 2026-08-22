from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..services import event_service as svc
from ..security import require_operator

router = APIRouter(prefix="/api/events", tags=["events"])


class ReviewRequest(BaseModel):
    action: str = Field(min_length=1, max_length=20)
    note: str | None = Field(default=None, max_length=500)


@router.get("")
def events(limit: int = 100, offset: int = 0, event_type: str | None = None, severity: str | None = None):
    return svc.list_events(limit=limit, offset=offset, event_type=event_type, severity=severity)


@router.get("/summary")
def summary():
    return svc.event_summary()


@router.get("/{event_id}")
def event(event_id: int):
    return svc.get_event(event_id)


@router.get("/{event_id}/snapshot")
def snapshot(event_id: int):
    return svc.snapshot_response(event_id)


@router.post("/{event_id}/review", dependencies=[Depends(require_operator)])
def review(event_id: int, payload: ReviewRequest):
    return svc.review_event(event_id, payload.action, payload.note)
