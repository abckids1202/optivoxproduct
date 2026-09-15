from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..services import people_service as svc
from ..security import require_admin, require_operator

router = APIRouter(prefix="/api/people", tags=["people"])


class PersonUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    role: str | None = Field(default=None, max_length=120)
    className: str | None = Field(default=None, max_length=120)
    studentId: str | None = Field(default=None, max_length=120)
    subjects: list[str] | str | None = None
    notes: str | None = Field(default=None, max_length=500)


@router.get("")
def people(actor: str = Depends(require_operator)):
    return svc.list_people()


@router.get("/{person_id}")
def person(person_id: int, actor: str = Depends(require_operator)):
    return svc.get_person(person_id)


@router.patch("/{person_id}")
def update(person_id: int, payload: PersonUpdateRequest, actor: str = Depends(require_admin)):
    return svc.update_person(person_id, payload.dict(exclude_none=True), actor_id=actor)


@router.post("/{person_id}/disable")
def disable(person_id: int, actor: str = Depends(require_admin)):
    return svc.set_enabled(person_id, False, actor_id=actor)


@router.post("/{person_id}/enable")
def enable(person_id: int, actor: str = Depends(require_admin)):
    return svc.set_enabled(person_id, True, actor_id=actor)
