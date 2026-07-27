from fastapi import APIRouter

from ..services import people_service as svc

router = APIRouter(prefix="/api/people", tags=["people"])


@router.get("")
def people():
    return svc.list_people()


@router.get("/{person_id}")
def person(person_id: int):
    return svc.get_person(person_id)


@router.patch("/{person_id}")
def update(person_id: int, payload: dict):
    return svc.update_person(person_id, payload)


@router.post("/{person_id}/disable")
def disable(person_id: int):
    return svc.set_enabled(person_id, False)


@router.post("/{person_id}/enable")
def enable(person_id: int):
    return svc.set_enabled(person_id, True)

