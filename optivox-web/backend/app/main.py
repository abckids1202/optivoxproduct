"""Flask application factory for the OptiVox JSON backend."""
from flask import Flask, jsonify
from flask_cors import CORS

from app.api.routes.auth import auth_api
from app.api.routes.vision import vision_api
from app.config import Config


def create_app(start_bridge=None):
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(
        app,
        origins=Config.CORS_ORIGINS,
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    app.register_blueprint(auth_api, url_prefix="/api/v1")
    app.register_blueprint(vision_api, url_prefix="/api/v1")

    # Legacy /api routes keep older dashboard integrations working while the React app uses /api/v1.
    app.register_blueprint(auth_api, url_prefix="/api", name="auth_api_legacy")
    app.register_blueprint(vision_api, url_prefix="/api", name="vision_api_legacy")

    @app.get("/")
    def root():
        return jsonify({
            "service": "optivox-backend",
            "status": "running",
            "docs": "Use the React frontend in ../frontend for browser pages.",
        })

    should_start = app.config["ENABLE_BRIDGE"] if start_bridge is None else start_bridge
    if should_start:
        from app.core.vision_bridge import VisionBridge

        bridge = VisionBridge(Config)
        bridge.start()
        app.config["_BRIDGE"] = bridge
        app.logger.info("Vision bridge started.")
    else:
        app.config["_BRIDGE"] = None

    return app


app = create_app()
