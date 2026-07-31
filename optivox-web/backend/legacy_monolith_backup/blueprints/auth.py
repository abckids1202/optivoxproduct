"""Session auth for the organization dashboard."""
from functools import wraps

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

auth = Blueprint("auth", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        cfg = current_app.config
        if username == cfg["ORG_USERNAME"] and password == cfg["ORG_PASSWORD"]:
            session["authed"] = True
            session["org"] = username
            return redirect(url_for("dashboard.index"))
        return render_template("login.html", error="Invalid credentials", active=None, year=2026)

    if session.get("authed"):
        return redirect(url_for("dashboard.index"))
    return render_template("login.html", active=None, year=2026)


@auth.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("public.index"))
