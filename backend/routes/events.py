from __future__ import annotations

from fastapi import APIRouter

from ..services import event_service as svc

router = APIRouter(prefix="/api/events", tags=["events"])


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
