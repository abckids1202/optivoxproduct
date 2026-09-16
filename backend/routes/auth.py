from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field

from ..config import COOKIE_SAMESITE, COOKIE_SECURE, CSRF_COOKIE_NAME, SESSION_COOKIE_NAME
from ..security import enforce_login_rate_limit, require_operator
from ..services import auth_service

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=256)


@router.post("/login")
def login(request: Request, payload: LoginRequest):
    enforce_login_rate_limit(request)
    result = auth_service.login(
        payload.username,
        payload.password,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    response = Response(
        content=json.dumps({"expires_at": result["expires_at"], "user": result["user"]}),
        media_type="application/json",
    )
    max_age = 8 * 60 * 60
    response.set_cookie(SESSION_COOKIE_NAME, result["access_token"], max_age=max_age, httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path="/")
    response.set_cookie(CSRF_COOKIE_NAME, result["csrf_token"], max_age=max_age, httponly=False, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path="/")
    return response


@router.get("/me")
def me(actor: str = Depends(require_operator)):
    return {"actor": actor}


@router.post("/logout")
def logout(request: Request, response: Response, authorization: str | None = Header(default=None), actor: str = Depends(require_operator)):
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else request.cookies.get(SESSION_COOKIE_NAME)
    auth_service.logout(token)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    return {"status": "logged_out", "actor": actor}


@router.post("/logout-all")
def logout_all(request: Request, response: Response, actor: str = Depends(require_operator)):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    session = auth_service.resolve_session(token)
    revoked = auth_service.logout_all(int(session["id"])) if session else 0
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    return {"status": "logged_out_everywhere", "actor": actor, "revoked_sessions": revoked}
