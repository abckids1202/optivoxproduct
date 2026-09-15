from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from ..security import enforce_login_rate_limit, require_operator
from ..services import auth_service

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=256)


@router.post("/login")
def login(request: Request, payload: LoginRequest):
    enforce_login_rate_limit(request)
    return auth_service.login(payload.username, payload.password)


@router.get("/me")
def me(actor: str = Depends(require_operator)):
    return {"actor": actor}


@router.post("/logout")
def logout(authorization: str | None = Header(default=None), actor: str = Depends(require_operator)):
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
    auth_service.logout(token)
    return {"status": "logged_out", "actor": actor}
