"""JSON API and MJPEG video feed."""
from flask import Blueprint, Response, current_app, jsonify, request

from core.db import WebDB
from core.stream import buffer, mjpeg_generator
from .auth import login_required

api = Blueprint("api", __name__, url_prefix="/api")


def _db():
    return WebDB(current_app.config["DATABASE_FILE"])


def _bridge():
    return current_app.config.get("_BRIDGE")


def _active_strangers():
    bridge = _bridge()
    if bridge:
        return len(bridge.live_state().get("active_strangers", []))
    return 0


@api.route("/video_feed")
@login_required
def video_feed():
    return Response(
        mjpeg_generator(fps=current_app.config.get("STREAM_FPS", 20)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@api.route("/stats")
@login_required
def stats():
    return jsonify(_db().stats(active_strangers=_active_strangers()))


@api.route("/events")
@login_required
def events():
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({"events": _db().recent_events(limit=limit)})


@api.route("/attendance")
@login_required
def attendance():
    return jsonify({"records": _db().attendance_today()})


@api.route("/people")
@login_required
def people():
    return jsonify({"people": _db().people()})


@api.route("/live_state")
@login_required
def live_state():
    bridge = _bridge()
    if not bridge:
        return jsonify({"bridge": {"mode": "disabled"}, "stream_meta": buffer.meta()})
    return jsonify(bridge.live_state())


@api.route("/toggle", methods=["POST"])
@login_required
def toggle():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, msg = bridge.set_toggle(data.get("name"), data.get("value"))
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 400


@api.route("/enroll_visible_face", methods=["POST"])
@login_required
def enroll_visible_face():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, msg = bridge.enroll_visible_face(data.get("name"))
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 400


@api.route("/manual_clock_in", methods=["POST"])
@login_required
def manual_clock_in():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, result = bridge.manual_clock_in(data.get("name"))
    return jsonify({"ok": ok, "result": result}), 200 if ok else 400


@api.route("/manual_clock_out", methods=["POST"])
@login_required
def manual_clock_out():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "message": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, result = bridge.manual_clock_out(data.get("name"))
    return jsonify({"ok": ok, "result": result}), 200 if ok else 400


@api.route("/assistant/ask", methods=["POST"])
@login_required
def assistant_ask():
    bridge = _bridge()
    if not bridge:
        return jsonify({"ok": False, "answer": "bridge is disabled"}), 409
    data = request.get_json(silent=True) or {}
    ok, answer = bridge.ask_assistant(data.get("question"))
    return jsonify({"ok": ok, "answer": answer}), 200 if ok else 400


@api.route("/health")
def health():
    bridge = _bridge()
    return jsonify({
        "status": "ok",
        "bridge": bridge.status() if bridge else {"mode": "disabled"},
        "stream_meta": buffer.meta(),
    })