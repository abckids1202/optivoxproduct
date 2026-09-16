from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..services import operational_service as svc
from ..security import require_permission

router = APIRouter(prefix="/api/operations", tags=["operations"])


@router.get("/summary")
def operational_summary(actor: str = Depends(require_permission("operations.view"))):
    return svc.summary()


@router.get("/presence")
def presence(
    limit: int = Query(default=100, ge=1, le=500),
    status: str | None = Query(default=None, max_length=24),
    person_id: int | None = Query(default=None, ge=1),
    actor: str = Depends(require_permission("operations.view")),
):
    return svc.list_presence_sessions(limit=limit, status=status, person_id=person_id)


@router.get("/presence/{session_id}")
def presence_detail(session_id: int, actor: str = Depends(require_permission("operations.view"))):
    return svc.get_presence_session(session_id)


@router.get("/recognition-evidence")
def recognition_evidence(
    limit: int = Query(default=100, ge=1, le=500),
    session_id: int | None = None,
    entity_id: str | None = None,
    person_id: int | None = None,
    decision: str | None = Query(default=None, max_length=32),
    actor: str = Depends(require_permission("operations.view")),
):
    return svc.list_recognition_evidence(
        limit=limit,
        session_id=session_id,
        entity_id=entity_id,
        person_id=person_id,
        decision=decision,
    )


@router.get("/attendance-decisions")
def attendance_decisions(
    limit: int = Query(default=100, ge=1, le=500),
    entity_id: str | None = None,
    person_id: int | None = Query(default=None, ge=1),
    decision: str | None = Query(default=None, max_length=32),
    actor: str = Depends(require_permission("operations.view")),
):
    return svc.list_attendance_decisions(
        limit=limit,
        entity_id=entity_id,
        person_id=person_id,
        decision=decision,
    )
