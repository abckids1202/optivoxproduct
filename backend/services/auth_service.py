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


_PBKDF2_ITERATIONS = 240_000
_SESSION_HOURS = 8


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
            [username, hash_password(password), role],
        )


def login(username: str, password: str) -> dict[str, Any]:
    row = fetch_one(
        "select id, username, password_hash, role, active from platform_users where lower(username)=lower(?)",
        [username.strip()],
    )
    if not row or not row.get("active") or not verify_password(password, row.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail={"code": "INVALID_CREDENTIALS", "message": "Username or password is incorrect."})
    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(hours=_SESSION_HOURS)
    execute(
        "insert into platform_sessions (token_hash, user_id, expires_at) values (?, ?, ?)",
        [_token_hash(token), row["id"], expires.isoformat()],
    )
    return {
        "access_token": token,
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
        execute("delete from platform_sessions where token_hash=?", [_token_hash(token)])

