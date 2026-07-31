"""OptiVox Web application factory."""
from flask import Flask

from blueprints.api import api
from blueprints.auth import auth
from blueprints.dashboard import dashboard
from blueprints.public import public
from config import Config


def create_app(start_bridge=None):
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(public)
    app.register_blueprint(auth)
    app.register_blueprint(dashboard)
    app.register_blueprint(api)

    should_start = app.config["ENABLE_BRIDGE"] if start_bridge is None else start_bridge
    if should_start:
        from core.vision_bridge import VisionBridge

        bridge = VisionBridge(Config)
        bridge.start()
        app.config["_BRIDGE"] = bridge
        app.logger.info("Vision bridge started.")
    else:
        app.config["_BRIDGE"] = None

    return app


app = create_app()