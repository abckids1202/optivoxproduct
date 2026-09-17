from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..config import COMMAND_RESULTS_PATH, COMMANDS_PATH, ENROLLMENT_STATUS_PATH, TIMEZONE
from ..database import execute, fetch_one

ALLOWED_COMMANDS = {
    "save_snapshot",
    "manual_clock_in",
    "manual_clock_out",
    "start_enrollment",
    "confirm_enrollment",
    "register_visible_unknown",
    "cancel_enrollment",
    "retrain_person",
    "disable_person",
    "merge_people",
    "delete_person",
    "test_alert",
    "reset_demo_data",
}

ADMIN_COMMANDS = {
    "retrain_person",
    "disable_person",
    "merge_people",
    "delete_person",
}

COMMAND_PERMISSIONS = {
    "save_snapshot": "live.view",
    "manual_clock_in": "attendance.manual",
    "manual_clock_out": "attendance.manual",
    "start_enrollment": "biometric.enroll",
    "register_visible_unknown": "biometric.enroll",
    "cancel_enrollment": "biometric.enroll",
    "confirm_enrollment": "biometric.enroll",
    "retrain_person": "biometric.manage",
    "disable_person": "biometric.manage",
    "merge_people": "biometric.merge",
    "delete_person": "biometric.delete",
    "test_alert": "system.manage",
    "reset_demo_data": "system.manage",
}

_COMMAND_LOCK = threading.Lock()


def atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        # Do not turn a full/unavailable disk into a misleading successful
        # command submission, and do not leave a partial command artifact.
        raise HTTPException(
            status_code=507,
            detail={"code": "COMMAND_STORAGE_UNAVAILABLE", "message": "The local command queue cannot be written."},
        ) from exc
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def read_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("commands", data.get("results", []))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def create_command(command_type: str, payload: dict[str, Any] | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
    if command_type not in ALLOWED_COMMANDS:
        raise HTTPException(status_code=400, detail={"code": "INVALID_COMMAND", "message": "Unsupported command type."})
    key = (idempotency_key or "").strip()[:128] or None
    with _COMMAND_LOCK:
        if key:
            existing = fetch_one("select command_id from command_idempotency where idempotency_key=?", [key])
            if existing:
                try:
                    return command_status(existing["command_id"])
                except HTTPException:
                    # The JSON bridge intentionally retains only a short
                    # pending/processing window; allow a reused key after its
                    # original command has aged out.
                    execute("delete from command_idempotency where idempotency_key=?", [key])
        command = {
            "id": f"cmd_{datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}",
            "type": command_type,
            "created_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
            "status": "pending",
            "payload": payload or {},
            "idempotency_key": key,
        }
        commands = read_list(COMMANDS_PATH)
        commands = [c for c in commands if c.get("status") in {"pending", "processing"}][-50:]
        commands.append(command)
        atomic_write(COMMANDS_PATH, {"commands": commands})
        if key:
            execute("insert or ignore into command_idempotency (idempotency_key, command_id) values (?, ?)", [key, command["id"]])
        return command


def command_status(command_id: str) -> dict[str, Any]:
    for item in read_list(COMMAND_RESULTS_PATH) + read_list(COMMANDS_PATH):
        if item.get("id") == command_id:
            return item
    raise HTTPException(status_code=404, detail={"code": "COMMAND_NOT_FOUND", "message": "Command was not found."})


def enrollment_status() -> dict[str, Any]:
    if not ENROLLMENT_STATUS_PATH.exists():
        return {"stage": "idle", "message": "No enrollment in progress."}
    try:
        return json.loads(ENROLLMENT_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"stage": "unknown", "message": "Enrollment status file is unreadable."}
