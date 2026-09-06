from __future__ import annotations

from fastapi import APIRouter, Query

from ..services import operational_service as svc

router = APIRouter(prefix="/api/operations", tags=["operations"])


@router.get("/summary")
def operational_summary():
    return svc.summary()


@router.get("/presence")
def presence(
    limit: int = Query(default=100, ge=1, le=500),
    status: str | None = Query(default=None, max_length=24),
    person_id: int | None = Query(default=None, ge=1),
):
    return svc.list_presence_sessions(limit=limit, status=status, person_id=person_id)


@router.get("/presence/{session_id}")
def presence_detail(session_id: int):
    return svc.get_presence_session(session_id)


@router.get("/recognition-evidence")
def recognition_evidence(
    limit: int = Query(default=100, ge=1, le=500),
    session_id: int | None = None,
    entity_id: str | None = None,
    person_id: int | None = None,
    decision: str | None = Query(default=None, max_length=32),
):
    return svc.list_recognition_evidence(
        limit=limit,
        session_id=session_id,
        entity_id=entity_id,
        person_id=person_id,
        decision=decision,
    )
