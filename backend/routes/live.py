import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services import runtime_service
from ..services.analytics_service import attendance as attendance_analytics, security as security_analytics
from ..services.attendance_service import attendance_summary, today_attendance
from ..services.event_service import event_summary, list_events
from ..services.incident_service import list_incidents
from ..services.people_service import list_people

router = APIRouter(tags=["live"])


@router.get("/api/live/status")
def live_status():
    state = runtime_service.live_state()
    state["people"] = list_people()
    state["attendance"] = today_attendance()
    state["events"] = list_events(limit=8)
    state["incidents"] = list_incidents(limit=8)
    attendance = attendance_summary()
    state["summary"].update(attendance)
    state["summary"]["presentToday"] = attendance["present"]
    state["summary"]["openIncidents"] = sum(1 for incident in state["incidents"] if incident["status"] not in {"dismissed", "resolved"})
    state["summary"]["securityObservations"] = event_summary()["total"]
    state["summary"]["securityEvents"] = state["summary"]["openIncidents"]
    state["analytics"] = {
        **attendance_analytics(),
        **security_analytics(),
    }
    return state


@router.get("/api/live/detections")
def live_detections():
    return runtime_service.live_detections()


@router.get("/api/live/frame")
def live_frame():
    return runtime_service.frame_response()


@router.get("/api/live/events")
def live_events():
    return list_events(limit=20)


@router.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    last_payload = ""
    try:
        await websocket.send_json({"type": "connected", "timestamp": runtime_service.now_iso()})
        while True:
            payload = {
                "type": "live_state",
                "state": live_status(),
                "timestamp": runtime_service.now_iso(),
            }
            encoded = json.dumps(payload, sort_keys=True, default=str)
            if encoded != last_payload:
                await websocket.send_json(payload)
                last_payload = encoded
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
