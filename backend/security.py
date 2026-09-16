from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, WebSocket, WebSocketException, status

from .config import CSRF_COOKIE_NAME, SESSION_COOKIE_NAME, configured_auth_methods, frontend_origin_allowed, loopback_bypass_allowed
from .services.auth_service import resolve_session, session_status


_RATE_LIMIT = 120
_RATE_WINDOW_SECONDS = 60
_LOGIN_RATE_LIMIT = 10
_rate_lock = threading.Lock()
_rate_history: dict[str, deque[float]] = defaultdict(deque)
_login_history: dict[str, deque[float]] = defaultdict(deque)

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"live.view", "attendance.view", "people.view", "security.view", "analytics.view", "system.view", "operations.view"}),
    "operator": frozenset({
        "live.view", "attendance.view", "attendance.manual", "people.view", "security.view",
        "evidence.view", "analytics.view", "system.view", "operations.view", "commands.execute",
    }),
    "security-reviewer": frozenset({"live.view", "security.view", "security.review", "evidence.view", "operations.view", "system.view"}),
    "attendance-admin": frozenset({"live.view", "attendance.view", "attendance.manual", "attendance.manage", "attendance.export", "people.view", "analytics.view", "operations.view", "system.view", "commands.execute"}),
    "biometric-admin": frozenset({"live.view", "people.view", "biometric.enroll", "biometric.manage", "biometric.merge", "biometric.delete", "operations.view", "system.view", "commands.execute"}),
    "system-admin": frozenset({"*"}),
    # Existing local/API deployments may still contain this legacy role.
    "admin": frozenset({"*"}),
}


def _configured_keys() -> tuple[str | None, str | None]:
    return os.getenv("OPTIVOX_API_KEY") or os.getenv("OPTIVOX_OPERATOR_KEY"), os.getenv("OPTIVOX_ADMIN_KEY")


def _supplied_key(x_optivox_key: str | None, authorization: str | None) -> str | None:
    if x_optivox_key:
        return x_optivox_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _supplied_credential(request: Request, x_optivox_key: str | None, authorization: str | None) -> tuple[str | None, bool]:
    header_credential = _supplied_key(x_optivox_key, authorization)
    if header_credential:
        return header_credential, False
    cookie_credential = request.cookies.get(SESSION_COOKIE_NAME)
    return (cookie_credential, True) if cookie_credential else (None, False)


def has_permission(role: str | None, permission: str) -> bool:
    normalized = str(role or "").strip().casefold()
    if normalized.startswith("local-"):
        return True
    permissions = ROLE_PERMISSIONS.get(normalized, frozenset())
    return "*" in permissions or permission in permissions


def _require_permission(role: str, permission: str | None) -> None:
    if permission and not has_permission(role, permission):
        raise HTTPException(status_code=403, detail={"code": "PERMISSION_REQUIRED", "message": f"Permission required: {permission}."})


def _record_cyber_event(event_type: str, request: Request, *, actor_id: str | None = None,
                        actor_type: str | None = None, details: dict | None = None,
                        dedupe_key: str | None = None) -> None:
    """Security telemetry must never make the protected request fail open/closed."""
    try:
        from .services.cybersecurity_service import record_cyber_event
        record_cyber_event(
            event_type,
            source="authorization" if event_type != "EXPIRED_TOKEN" else "authentication",
            actor_id=actor_id,
            actor_type=actor_type or "system",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"method": request.method, "path": request.url.path, **(details or {})},
            dedupe_key=dedupe_key,
        )
    except Exception:
        # Observability must not become an authentication bypass or an outage.
        return


def _record_websocket_event(event_type: str, websocket: WebSocket, *, actor_id: str | None = None,
                            actor_type: str | None = None, details: dict | None = None) -> None:
    try:
        from .services.cybersecurity_service import record_cyber_event
        host = websocket.client.host if websocket.client else None
        record_cyber_event(
            event_type,
            source="websocket",
            actor_id=actor_id,
            actor_type=actor_type or "system",
            ip_address=host,
            user_agent=websocket.headers.get("user-agent"),
            details={"path": websocket.url.path, **(details or {})},
            dedupe_key=f"websocket:{event_type}:{host or 'unknown'}:{websocket.url.path}:{int(time.time() // 60)}",
        )
    except Exception:
        return


def _validate_csrf(request: Request, cookie_auth: bool) -> None:
    if not cookie_auth or request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    supplied_token = request.headers.get("X-CSRF-Token")
    if not cookie_token or not supplied_token or not hmac.compare_digest(cookie_token, supplied_token):
        raise HTTPException(status_code=403, detail={"code": "CSRF_TOKEN_REQUIRED", "message": "A valid CSRF token is required for this session action."})


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _guard(request: Request, x_optivox_key: str | None, authorization: str | None,
           admin: bool = False, required_permission: str | None = None) -> str:
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
    supplied, cookie_auth = _supplied_credential(request, x_optivox_key, authorization)
    session = resolve_session(supplied)
    if session:
        _validate_csrf(request, cookie_auth)
        permission = required_permission or ("system.admin" if admin else None)
        if permission and not has_permission(str(session.get("role") or ""), permission):
            _record_cyber_event(
                "UNAUTHORIZED_COMMAND" if request.url.path.rstrip("/") == "/api/commands" else "WRONG_ROLE_ACCESS",
                request,
                actor_id=str(session.get("id") or session.get("username") or "unknown"),
                actor_type="user",
                details={"username": session.get("username"), "role": session.get("role"), "required_permission": permission},
                dedupe_key=f"wrong-role:{session.get('id')}:{request.url.path}:{permission}:{int(time.time() // 60)}",
            )
            _require_permission(str(session.get("role") or ""), permission)
        return str(session.get("username") or "session-user")
    if supplied and session_status(supplied) == "expired":
        _record_cyber_event(
            "EXPIRED_TOKEN", request,
            details={"credential_type": "session_cookie" if cookie_auth else "authorization_header"},
            dedupe_key=f"expired-token:{request.client.host if request.client else 'unknown'}:{request.url.path}:{int(time.time() // 60)}",
        )
    # An operator key must never grant administrator access just because a
    # separate administrator key was omitted. Durable admin users remain a
    # supported alternative through the session branch above.
    expected = admin_key if admin else (operator_key or admin_key)
    if expected:
        if not supplied or not hmac.compare_digest(supplied, expected):
            if request.url.path.rstrip("/") == "/api/commands":
                _record_cyber_event("UNAUTHORIZED_COMMAND", request, details={"credential_type": "api_key"})
            raise HTTPException(status_code=401, detail={"code": "INVALID_API_KEY", "message": "A valid OptiVox API key is required."})
        role = "system-admin" if admin or (admin_key and hmac.compare_digest(supplied, admin_key)) else "operator"
        _require_permission(role, required_permission or ("system.admin" if admin else None))
        return "api-key-admin" if role == "system-admin" else "api-key-operator"
    if loopback_bypass_allowed() and _is_loopback(request):
        _validate_csrf(request, False)
        _require_permission("local-admin" if admin else "local-operator", required_permission or ("system.admin" if admin else None))
        return "local-admin" if admin else "local-operator"
    if not any(configured_auth_methods().values()):
        raise HTTPException(status_code=503, detail={"code": "AUTH_NOT_CONFIGURED", "message": "Configure an API key or durable user before using pilot or production mode."})
    raise HTTPException(status_code=401, detail={"code": "AUTHENTICATION_REQUIRED", "message": "A valid authenticated session is required."})


def enforce_login_rate_limit(request: Request) -> None:
    """Limit password attempts independently from normal operator traffic."""
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        history = _login_history[client]
        while history and now - history[0] > _RATE_WINDOW_SECONDS:
            history.popleft()
        if len(history) >= _LOGIN_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={"code": "LOGIN_RATE_LIMITED", "message": "Too many sign-in attempts. Try again shortly."},
                headers={"Retry-After": "60"},
            )
        history.append(now)


def require_operator(
    request: Request,
    x_optivox_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    return _guard(request, x_optivox_key, authorization)


def require_admin(
    request: Request,
    x_optivox_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    return _guard(request, x_optivox_key, authorization, admin=True, required_permission="system.admin")


def require_permission(permission: str):
    """Create a FastAPI dependency enforcing one named permission."""
    def dependency(
        request: Request,
        x_optivox_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> str:
        return _guard(request, x_optivox_key, authorization, required_permission=permission)
    dependency.__name__ = f"require_{permission.replace('.', '_')}"
    return dependency


def require_websocket_operator(websocket: WebSocket) -> str:
    """Protect the live stream before accepting a websocket connection."""
    # HTTP CORS does not protect WebSocket upgrades. Validate the browser
    # origin separately while allowing origin-less trusted native clients.
    if not frontend_origin_allowed(websocket.headers.get("origin")):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    operator_key, admin_key = _configured_keys()
    supplied = _supplied_key(
        websocket.headers.get("x-optivox-key"),
        websocket.headers.get("authorization"),
    )
    if not supplied:
        supplied = websocket.cookies.get("optivox_session")
    session = resolve_session(supplied)
    if session:
        if not has_permission(str(session.get("role") or ""), "live.view"):
            _record_websocket_event(
                "WRONG_ROLE_ACCESS", websocket,
                actor_id=str(session.get("id") or session.get("username") or "unknown"),
                actor_type="user", details={"username": session.get("username"), "role": session.get("role"), "required_permission": "live.view"},
            )
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        return str(session.get("username") or "session-user")
    if supplied and session_status(supplied) == "expired":
        _record_websocket_event("EXPIRED_TOKEN", websocket, details={"credential_type": "session_cookie"})
    expected = operator_key or admin_key
    if expected and (not supplied or not hmac.compare_digest(supplied, expected)):
        _record_websocket_event("WRONG_ROLE_ACCESS", websocket, details={"credential_type": "api_key"})
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    if not expected:
        if not loopback_bypass_allowed():
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        host = websocket.client.host if websocket.client else ""
        if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    return "api-key-operator" if expected else "local-operator"
