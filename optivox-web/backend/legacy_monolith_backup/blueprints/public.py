"""Public marketing site routes."""
from datetime import datetime

from flask import Blueprint, render_template

public = Blueprint("public", __name__)


def _ctx(active):
    return {"active": active, "year": datetime.now().year}


@public.route("/")
def index():
    return render_template("index.html", **_ctx("home"))


@public.route("/features")
def features():
    return render_template("features.html", **_ctx("features"))


@public.route("/demo")
def demo():
    return render_template("demo.html", **_ctx("demo"))


@public.route("/about")
def about():
    return render_template("about.html", **_ctx("about"))


@public.route("/faq")
def faq():
    return render_template("faq.html", **_ctx("faq"))