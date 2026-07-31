"""Session authentication API for the organization dashboard."""
from functools import wraps

from flask import Blueprint, current_app, jsonify, request, session

auth_api = Blueprint("auth_api", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            return jsonify({
                "authenticated": False,
                "message": "Authentication required",
            }), 401
        return view(*args, **kwargs)

    return wrapped


@auth_api.post("/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", ""))
    password = str(data.get("password", ""))
    cfg = current_app.config

    if username == cfg["ORG_USERNAME"] and password == cfg["ORG_PASSWORD"]:
        session["authed"] = True
        session["org"] = username
        return jsonify({
            "authenticated": True,
            "org": username,
        })

    return jsonify({
        "authenticated": False,
        "message": "Invalid credentials",
    }), 401


@auth_api.route("/auth/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return jsonify({"authenticated": False})


@auth_api.get("/auth/me")
@login_required
def me():
    return jsonify({
        "authenticated": True,
        "org": session.get("org"),
    })
