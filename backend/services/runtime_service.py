from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..config import CAPABILITY_PATH, DEVICE_ID, HEARTBEAT_PATH, LATEST_FRAME_PATH, LIVE_STATE_PATH, PERFORMANCE_SUMMARY_PATH, TIMEZONE


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


def capability_state() -> dict[str, Any]:
    capabilities = read_json(CAPABILITY_PATH, {}) or {}
    if not capabilities:
        return {"available": False, "capabilities": {}}
    return {
        "available": True,
        "runtime_id": capabilities.get("runtime_id"),
        "runtime_version": capabilities.get("runtime_version"),
        "pid": capabilities.get("pid"),
        "started_at": capabilities.get("started_at"),
        "generated_at": capabilities.get("generated_at"),
        "capabilities": capabilities.get("capabilities", {}),
    }


def live_state() -> dict[str, Any]:
    hb = heartbeat_state()
    capabilities = capability_state()
    state = read_json(LIVE_STATE_PATH, {}) or {}
    engine = state.get("engine", {})
    camera = state.get("camera", {})
    presence = state.get("presence", {})
    objects = state.get("objects", [])
    security = state.get("security", {})
    performance = state.get("performance", {})
    performance_summary = read_json(PERFORMANCE_SUMMARY_PATH, {}) or {}
    correlation = state.get("correlation", {})
    return {
        "generatedAt": now_iso(),
        "device": {"id": DEVICE_ID, "type": "edge-agent", "biometric_owner": "local_engine"},
        "runtime": {
            "id": capabilities.get("runtime_id") or (hb.get("heartbeat") or {}).get("runtime_id"),
            "version": capabilities.get("runtime_version") or (hb.get("heartbeat") or {}).get("runtime_version"),
            "capabilities": capabilities.get("capabilities", {}),
            "capability_available": capabilities.get("available", False),
            "heartbeat_age_seconds": hb.get("age_seconds"),
        },
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
        "performance": performance,
        "performanceSummary": performance_summary,
        "correlation": correlation,
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


def performance_report() -> dict[str, Any]:
    """Return measured values when a fresh benchmark exists, otherwise explicit unknowns."""
    report = read_json(PERFORMANCE_SUMMARY_PATH, {}) or {}
    required = {
        "capture_fps": None,
        "inference_fps": None,
        "display_fps": None,
        "face_detection_latency_ms": None,
        "recognition_latency_ms": None,
        "yolo_latency_ms": None,
        "pose_latency_ms": None,
        "frame_age_p95_ms": None,
        "end_to_end_latency_ms": latency.get("end_to_end_p95"),
        "cpu_percent": None,
        "gpu_percent": None,
        "vram_used_mb": None,
        "recognition_attempts_per_second": None,
        "identity_confirmation_ms": None,
        "frames_replaced": None,
    }
    latency = report.get("latency_ms", {}) if isinstance(report.get("latency_ms", {}), dict) else {}
    vision = report.get("vision", {}) if isinstance(report.get("vision", {}), dict) else {}
    models = vision.get("models", {}) if isinstance(vision.get("models", {}), dict) else {}
    resources = report.get("resource", {}) if isinstance(report.get("resource", {}), dict) else {}
    matching = models.get("identity_matching", {}) if isinstance(models.get("identity_matching", {}), dict) else {}
    identity_timing = vision.get("identity_timing", {}) if isinstance(vision.get("identity_timing", {}), dict) else {}
    values = {**required, **report}
    values.update({
        "face_detection_latency_ms": (models.get("face_detection") or {}).get("average_latency_ms"),
        "recognition_latency_ms": matching.get("average_latency_ms"),
        "yolo_latency_ms": (models.get("yolo") or {}).get("average_latency_ms"),
        "pose_latency_ms": (models.get("pose") or {}).get("average_latency_ms"),
        "frame_age_p95_ms": latency.get("frame_age_p95"),
        "cpu_percent": resources.get("cpu_percent"),
        "gpu_percent": resources.get("gpu_percent"),
        "vram_used_mb": resources.get("vram_used_mb"),
        "recognition_attempts_per_second": matching.get("calls_per_second"),
        "identity_confirmation_ms": round(float(identity_timing.get("mean_time_to_confirm_sec")) * 1000, 2) if identity_timing.get("mean_time_to_confirm_sec") is not None else None,
        "frames_replaced": (report.get("counters") or {}).get("frames_replaced"),
        "end_to_end_latency_ms": None,
    })
    ended_at = (report.get("benchmark") or {}).get("ended_at")
    age_seconds = None
    if ended_at:
        try:
            parsed = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            pass
    values["measurement_status"] = "NOT_MEASURED" if not report else ("FRESH" if age_seconds is not None and age_seconds <= 86400 else "STALE")
    values["benchmark_age_seconds"] = round(age_seconds, 1) if age_seconds is not None else None
    values["benchmark"] = report.get("benchmark", {})
    values["source"] = str(PERFORMANCE_SUMMARY_PATH)
    return values


def live_detections() -> dict[str, Any]:
    state = read_json(LIVE_STATE_PATH, {}) or {}
    capabilities = capability_state()
    return {
        "runtime": {
            "id": capabilities.get("runtime_id") or state.get("runtime_id"),
            "version": capabilities.get("runtime_version") or state.get("runtime_version"),
            "capabilities": capabilities.get("capabilities", {}),
        },
        "presence": state.get("presence", {"registered": [], "unknown": []}),
        "objects": state.get("objects", []),
        "recent_events": state.get("recent_events", []),
        "security": state.get("security", {}),
        "performance": state.get("performance", {}),
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
            "entityId": p.get("entity_id"),
            "trackId": p.get("track_id"),
            "label": p.get("name", "Registered person"),
            "type": "registered",
            "confidence": p.get("confidence", 0),
            "identityState": p.get("identity_state", "UNRESOLVED"),
            "identityAgeMs": p.get("identity_age_ms"),
            "attendance": p.get("attendance_status", "present"),
            "visibleFor": format_duration(p.get("visible_seconds")),
            "note": "Registered attendance",
        })
    for p in presence.get("unknown", []):
        out.append({
            "id": f"unknown-{p.get('track_id')}",
            "entityId": p.get("entity_id"),
            "trackId": p.get("track_id"),
            "label": p.get("temporary_name", "Unknown person"),
            "type": "spoof" if p.get("spoof_status") == "suspect" else "unknown",
            "confidence": p.get("confidence", 0.5),
            "identityState": p.get("identity_state", "UNRESOLVED"),
            "identityAgeMs": p.get("identity_age_ms"),
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
