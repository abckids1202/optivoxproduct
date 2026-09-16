from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services import people_service as svc
from ..security import require_permission
from ..services.cybersecurity_service import record_cyber_event

router = APIRouter(prefix="/api/people", tags=["people"])


class PersonUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    role: str | None = Field(default=None, max_length=120)
    className: str | None = Field(default=None, max_length=120)
    studentId: str | None = Field(default=None, max_length=120)
    subjects: list[str] | str | None = None
    notes: str | None = Field(default=None, max_length=500)


@router.get("")
def people(actor: str = Depends(require_permission("people.view"))):
    return svc.list_people()


@router.get("/{person_id}")
def person(person_id: int, actor: str = Depends(require_permission("people.view"))):
    return svc.get_person(person_id)


@router.patch("/{person_id}")
def update(person_id: int, payload: PersonUpdateRequest, request: Request, actor: str = Depends(require_permission("people.manage"))):
    record_cyber_event("CONFIGURATION_CHANGE", source="people_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details={"resource": "person_profile", "person_id": person_id, "fields": sorted(payload.dict(exclude_none=True).keys())})
    return svc.update_person(person_id, payload.dict(exclude_none=True), actor_id=actor)


@router.post("/{person_id}/disable")
def disable(person_id: int, request: Request, actor: str = Depends(require_permission("biometric.manage"))):
    record_cyber_event("BIOMETRIC_CHANGE", source="people_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details={"operation": "disable", "person_id": person_id})
    return svc.set_enabled(person_id, False, actor_id=actor)


@router.post("/{person_id}/enable")
def enable(person_id: int, request: Request, actor: str = Depends(require_permission("biometric.manage"))):
    record_cyber_event("BIOMETRIC_CHANGE", source="people_api", actor_id=actor, actor_type="user", ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"), details={"operation": "enable", "person_id": person_id})
    return svc.set_enabled(person_id, True, actor_id=actor)
