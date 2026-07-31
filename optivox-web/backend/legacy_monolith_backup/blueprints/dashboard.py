"""Gated organization command center."""
from flask import Blueprint, render_template

from .auth import login_required

dashboard = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard.route("/")
@login_required
def index():
    return render_template("dashboard/index.html")
