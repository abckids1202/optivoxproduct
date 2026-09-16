from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from ..database import execute, fetch_one
from .audit_service import record_action


_PBKDF2_ITERATIONS = 240_000
_SESSION_HOURS = 8
_LOCKOUT_THRESHOLD = 8
_LOCKOUT_MINUTES = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str, salt: bytes | None = None) -> str:
    if not password or len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt, expected = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.urlsafe_b64decode(salt.encode("ascii")),
            int(iterations),
        )
        actual = base64.urlsafe_b64encode(digest).decode("ascii")
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError, UnicodeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _safe_actor(username: str) -> str:
    return str(username or "").strip()[:120] or "unknown"


def bootstrap_configured_users() -> None:
    """Create local users once from environment variables, never log passwords."""
    for prefix, role in (("ADMIN", "admin"), ("OPERATOR", "operator")):
        username = os.getenv(f"OPTIVOX_{prefix}_USERNAME", "").strip()
        password = os.getenv(f"OPTIVOX_{prefix}_PASSWORD", "")
        if not username or not password:
            continue
        existing = fetch_one("select id from platform_users where username=?", [username])
        if existing:
            continue
        execute(
            "insert into platform_users (username, password_hash, role) values (?, ?, ?)",
            [username, hash_password(password), "system-admin" if role == "admin" else role],
        )


def login(username: str, password: str, client_ip: str | None = None, user_agent: str | None = None) -> dict[str, Any]:
    username = _safe_actor(username)
    row = fetch_one(
        "select id, username, password_hash, role, active, failed_login_count, locked_until from platform_users where lower(username)=lower(?)",
        [username],
    )
    now = _now()
    if row and row.get("locked_until"):
        try:
            locked_until = datetime.fromisoformat(str(row["locked_until"]))
            if locked_until.tzinfo is None:
                locked_until = locked_until.replace(tzinfo=timezone.utc)
            if locked_until > now:
                record_action("auth.login.locked", "user", row.get("id"), {"username": username}, actor_type="system", actor_id=username)
                raise HTTPException(status_code=429, detail={"code": "ACCOUNT_TEMPORARILY_LOCKED", "message": "Account temporarily locked. Try again later."})
        except ValueError:
            pass
    if not row or not row.get("active") or not verify_password(password, row.get("password_hash") or ""):
        if row:
            failed = int(row.get("failed_login_count") or 0) + 1
            locked_until = None
            if failed >= _LOCKOUT_THRESHOLD:
                locked_until = (now + timedelta(minutes=_LOCKOUT_MINUTES)).isoformat()
                failed = 0
            execute(
                "update platform_users set failed_login_count=?, locked_until=?, updated_at=? where id=?",
                [failed, locked_until, now.isoformat(), row["id"]],
            )
        record_action("auth.login.failure", "user", row.get("id") if row else None, {"username": username}, actor_type="system", actor_id=username)
        raise HTTPException(status_code=401, detail={"code": "INVALID_CREDENTIALS", "message": "Username or password is incorrect."})

    # Re-authentication rotates the session family by revoking prior tokens.
    execute("delete from platform_sessions where user_id=?", [row["id"]])
    token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires = now + timedelta(hours=_SESSION_HOURS)
    execute(
        "insert into platform_sessions (token_hash, user_id, expires_at, csrf_token_hash, client_ip, user_agent) values (?, ?, ?, ?, ?, ?)",
        [_token_hash(token), row["id"], expires.isoformat(), _token_hash(csrf_token), (client_ip or "")[:128], (user_agent or "")[:300]],
    )
    execute(
        "update platform_users set failed_login_count=0, locked_until=null, last_login_at=?, last_login_ip=?, updated_at=? where id=?",
        [now.isoformat(), (client_ip or "")[:128], now.isoformat(), row["id"]],
    )
    record_action("auth.login.success", "user", row["id"], {"username": row["username"]}, actor_type="user", actor_id=row["username"])
    return {
        "access_token": token,
        "csrf_token": csrf_token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "user": {"id": row["id"], "username": row["username"], "role": row["role"]},
    }


def resolve_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    row = fetch_one(
        """
        select s.token_hash, s.expires_at, u.id as user_id, u.username, u.role
        from platform_sessions s join platform_users u on u.id=s.user_id
        where s.token_hash=? and s.expires_at>? and u.active=1
        """,
        [_token_hash(token), _now().isoformat()],
    )
    if not row:
        return None
    execute("update platform_sessions set last_seen_at=? where token_hash=?", [_now().isoformat(), row["token_hash"]])
    return {"id": row["user_id"], "username": row["username"], "role": row["role"]}


def logout(token: str | None) -> None:
    if token:
        session = fetch_one("select user_id from platform_sessions where token_hash=?", [_token_hash(token)])
        execute("delete from platform_sessions where token_hash=?", [_token_hash(token)])
        if session:
            record_action("auth.logout", "user", session["user_id"], {}, actor_type="user")


def logout_all(user_id: int) -> int:
    revoked = execute("delete from platform_sessions where user_id=?", [user_id])
    record_action("auth.logout_all", "user", user_id, {"revoked_sessions": revoked}, actor_type="user")
    return revoked
