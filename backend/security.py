from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request


_RATE_LIMIT = 120
_RATE_WINDOW_SECONDS = 60
_rate_lock = threading.Lock()
_rate_history: dict[str, deque[float]] = defaultdict(deque)


def _configured_keys() -> tuple[str | None, str | None]:
    return os.getenv("OPTIVOX_API_KEY") or os.getenv("OPTIVOX_OPERATOR_KEY"), os.getenv("OPTIVOX_ADMIN_KEY")


def _supplied_key(x_optivox_key: str | None, authorization: str | None) -> str | None:
    if x_optivox_key:
        return x_optivox_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _guard(request: Request, x_optivox_key: str | None, authorization: str | None, admin: bool = False) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        history = _rate_history[client]
        while history and now - history[0] > _RATE_WINDOW_SECONDS:
            history.popleft()
        if len(history) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail={"code": "RATE_LIMITED", "message": "Too many operator requests. Try again shortly."}, headers={"Retry-After": "60"})
        history.append(now)
    operator_key, admin_key = _configured_keys()
    supplied = _supplied_key(x_optivox_key, authorization)
    expected = admin_key if admin and admin_key else (operator_key or admin_key)
    if expected:
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail={"code": "INVALID_API_KEY", "message": "A valid OptiVox API key is required."})
        if admin and admin_key and not hmac.compare_digest(supplied, admin_key):
            raise HTTPException(status_code=403, detail={"code": "ADMIN_REQUIRED", "message": "This operation requires an administrator key."})
        return
    if not _is_loopback(request):
        raise HTTPException(status_code=503, detail={"code": "API_KEY_NOT_CONFIGURED", "message": "Configure OPTIVOX_API_KEY before exposing the backend beyond localhost."})


def require_operator(
    request: Request,
    x_optivox_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _guard(request, x_optivox_key, authorization)


def require_admin(
    request: Request,
    x_optivox_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    _guard(request, x_optivox_key, authorization, admin=True)
