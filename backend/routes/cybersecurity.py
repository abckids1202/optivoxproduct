from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..security import require_permission
from ..services import cybersecurity_service as svc


router = APIRouter(prefix="/api/cybersecurity", tags=["cybersecurity"])


class ReviewRequest(BaseModel):
    action: str = Field(min_length=1, max_length=24)
    note: str | None = Field(default=None, max_length=500)


class AssignmentRequest(BaseModel):
    assignee: str | None = Field(default=None, max_length=120)


@router.get("/summary")
def summary(actor: str = Depends(require_permission("security.view"))):
    return svc.cybersecurity_summary()


@router.get("/incidents")
def incidents(limit: int = 100, status: str | None = None, category: str | None = None,
              severity: int | None = None, actor: str = Depends(require_permission("security.view"))):
    return svc.list_cyber_incidents(limit=limit, status=status, category=category, severity=severity)


@router.get("/incidents/{incident_id}")
def incident(incident_id: int, actor: str = Depends(require_permission("security.view"))):
    return svc.get_cyber_incident(incident_id)


@router.post("/incidents/{incident_id}/review")
def review(incident_id: int, payload: ReviewRequest, actor: str = Depends(require_permission("security.review"))):
    return svc.review_cyber_incident(incident_id, payload.action, payload.note, actor_id=actor)


@router.post("/incidents/{incident_id}/assign")
def assign(incident_id: int, payload: AssignmentRequest, actor: str = Depends(require_permission("security.review"))):
    return svc.assign_cyber_incident(incident_id, payload.assignee, actor_id=actor)


@router.get("/events")
def events(limit: int = 200, event_type: str | None = None, actor: str = Depends(require_permission("security.view"))):
    return svc.list_cyber_events(limit=limit, event_type=event_type)


@router.get("/sessions")
def sessions(actor: str = Depends(require_permission("security.view"))):
    return svc.active_sessions()


@router.get("/auth-history")
def auth_history(limit: int = 100, actor: str = Depends(require_permission("security.view"))):
    return svc.authentication_history(limit)


@router.get("/admin-history")
def admin_history(limit: int = 100, actor: str = Depends(require_permission("security.view"))):
    return svc.administrative_history(limit)


@router.get("/integrity")
def integrity(actor: str = Depends(require_permission("security.view"))):
    return svc.integrity_failures()


@router.get("/alert-failures")
def alert_failures(actor: str = Depends(require_permission("security.view"))):
    return svc.alert_failures()
