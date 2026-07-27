from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..config import HEARTBEAT_PATH, LATEST_FRAME_PATH, LIVE_STATE_PATH, TIMEZONE


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def heartbeat_state() -> dict[str, Any]:
    hb = read_json(HEARTBEAT_PATH)
    if not hb:
        return {"status": "offline", "age_seconds": None, "heartbeat": None}
    timestamp = hb.get("timestamp")
    age = None
    try:
        age = time.time() - datetime.fromisoformat(timestamp).timestamp()
    except Exception:
        pass
    if age is None:
        status = "error"
    elif age <= 3:
        status = "online"
    elif age <= 10:
        status = "delayed"
    else:
        status = "offline"
    return {"status": status, "age_seconds": age, "heartbeat": hb}


def live_state() -> dict[str, Any]:
    hb = heartbeat_state()
    state = read_json(LIVE_STATE_PATH, {}) or {}
    engine = state.get("engine", {})
    camera = state.get("camera", {})
    presence = state.get("presence", {})
    objects = state.get("objects", [])
    security = state.get("security", {})
    return {
        "generatedAt": now_iso(),
        "localTime": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
        "connection": hb["status"],
        "engine": {
            "status": hb["status"].capitalize(),
            "camera": camera.get("status", "offline").capitalize(),
            "location": camera.get("location", "Class"),
            "fps": engine.get("fps") or (hb.get("heartbeat") or {}).get("fps") or 0,
            "uptime": format_duration(engine.get("uptime_seconds")),
            "mode": "Local AI Processing",
            "frameAge": frame_age(),
            "frameAvailable": LATEST_FRAME_PATH.exists(),
            "lastHeartbeat": (hb.get("heartbeat") or {}).get("timestamp"),
            "lastError": engine.get("last_error") or (hb.get("heartbeat") or {}).get("last_error"),
            "frameWidth": engine.get("frame_width"),
            "frameHeight": engine.get("frame_height"),
        },
        "security": {
            "level": security.get("level", "normal"),
            "message": security.get("message", "No active warning"),
        },
        "summary": {
            "presentToday": 0,
            "visibleNow": len(presence.get("registered", [])) + len(presence.get("unknown", [])),
            "unknownToday": len(presence.get("unknown", [])),
            "securityEvents": security.get("active_event_count", 0),
            "alertsSent": 0,
            "registeredPeople": len(presence.get("registered", [])),
        },
        "visiblePeople": normalize_people(presence),
        "objects": normalize_objects(objects),
        "events": state.get("recent_events", []),
    }


def live_detections() -> dict[str, Any]:
    state = read_json(LIVE_STATE_PATH, {}) or {}
    return {
        "presence": state.get("presence", {"registered": [], "unknown": []}),
        "objects": state.get("objects", []),
        "recent_events": state.get("recent_events", []),
        "security": state.get("security", {}),
        "timestamp": state.get("timestamp"),
    }


def frame_response() -> FileResponse:
    if not LATEST_FRAME_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "FRAME_UNAVAILABLE", "message": "No annotated frame has been published yet."},
        )
    return FileResponse(
        LATEST_FRAME_PATH,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


def frame_age() -> str:
    if not LATEST_FRAME_PATH.exists():
        return "unavailable"
    age = max(0, time.time() - LATEST_FRAME_PATH.stat().st_mtime)
    return f"{age:.1f}s"


def format_duration(seconds: Any) -> str:
    if not seconds:
        return "0s"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def normalize_people(presence: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in presence.get("registered", []):
        out.append({
            "id": f"registered-{p.get('track_id')}",
            "label": p.get("name", "Registered person"),
            "type": "registered",
            "confidence": p.get("confidence", 0),
            "attendance": p.get("attendance_status", "present"),
            "visibleFor": format_duration(p.get("visible_seconds")),
            "note": "Registered attendance",
        })
    for p in presence.get("unknown", []):
        out.append({
            "id": f"unknown-{p.get('track_id')}",
            "label": p.get("temporary_name", "Unknown person"),
            "type": "spoof" if p.get("spoof_status") == "suspect" else "unknown",
            "confidence": p.get("confidence", 0.5),
            "attendance": "Presence only",
            "visibleFor": format_duration(p.get("visible_seconds")),
            "note": "Unregistered person",
        })
    return out


def normalize_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": o.get("class_name") or o.get("name", "object"),
            "count": o.get("count", 1),
            "confidence": o.get("confidence", 0),
            "category": o.get("category", "object"),
        }
        for o in objects
    ]
