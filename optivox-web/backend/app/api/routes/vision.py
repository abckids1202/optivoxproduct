"""JSON API and MJPEG video feed for the OptiVox dashboard."""
from flask import Blueprint, Response, current_app, jsonify, request

from app.api.routes.auth import login_required
from app.core.db import WebDB
from app.core.stream import buffer, mjpeg_generator

vision_api = Blueprint("vision_api", __name__)


def _db():
    return WebDB(current_app.config["DATABASE_FILE"])


def _bridge():
    return current_app.config.get("_BRIDGE")


def _active_strangers():
    bridge = _bridge()
    if bridge:
        return len(bridge.live_state().get("active_strangers", []))
    return 0


@vision_api.route("/video-feed")
@vision_api.route("/video_feed")
@login_required
def video_feed():
    return Response(
        mjpeg_generator(fps=current_app.config.get("STREAM_FPS", 20)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@vision_api.get("/stats")
@login_required
def stats():
    return jsonify(_db().stats(active_strangers=_active_strangers()))


@vision_api.get("/events")
@login_required
def events():
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({"events": _db().recent_events(limit=limit)})


@vision_api.get("/attendance")
@login_required
def attendance():
    return jsonify({"records": _db().attendance_today()})


@vision_api.get("/people")
@login_required
def people():
    return jsonify({"people": _db().people()})


@vision_api.route("/live-state")
@vision_api.route("/live_state")
@login_required
def live_state():
    bridge = _bridge()
    if not bridge:
        return jsonify({"bridge": {"mode": "disabled"}, "stream_meta": buffer.meta()})
    return jsonify(bridge.live_state())


@vision_api.post("/toggle")
@login_required
def toggle():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, msg = bridge.set_toggle(data.get("name"), data.get("value"))
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 400


@vision_api.route("/enroll-visible-face", methods=["POST"])
@vision_api.route("/enroll_visible_face", methods=["POST"])
@login_required
def enroll_visible_face():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, msg = bridge.enroll_visible_face(data.get("name"))
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 400


@vision_api.route("/manual-clock-in", methods=["POST"])
@vision_api.route("/manual_clock_in", methods=["POST"])
@login_required
def manual_clock_in():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, result = bridge.manual_clock_in(data.get("name"))
    return jsonify({"ok": ok, "result": result}), 200 if ok else 400


@vision_api.route("/manual-clock-out", methods=["POST"])
@vision_api.route("/manual_clock_out", methods=["POST"])
@login_required
def manual_clock_out():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, result = bridge.manual_clock_out(data.get("name"))
    return jsonify({"ok": ok, "result": result}), 200 if ok else 400


@vision_api.post("/assistant/ask")
@login_required
def assistant_ask():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "answer": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, answer = bridge.ask_assistant(data.get("question"))
    return jsonify({"ok": ok, "answer": answer}), 200 if ok else 400


@vision_api.post("/contact")
def contact():
    data = request.get_json(silent=True) or {}
    if not data.get("name") or not data.get("email"):
        return jsonify({"ok": False, "message": "Name and email are required"}), 400
    current_app.logger.info("Contact request received: %s", {
        "name": data.get("name"),
        "email": data.get("email"),
        "org": data.get("org"),
        "type": data.get("type"),
    })
    return jsonify({
        "ok": True,
        "message": "Request received. We will be in touch.",
    })


@vision_api.get("/health")
def health():
    bridge = _bridge()
    return jsonify({
        "status": "healthy",
        "service": "optivox-backend",
        "bridge": bridge.status() if bridge else {"mode": "disabled"},
        "stream_meta": buffer.meta(),
    })
