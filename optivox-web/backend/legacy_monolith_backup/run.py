"""Development entry point for OptiVox Web."""
from app import create_app
from config import Config


if __name__ == "__main__":
    app = create_app(start_bridge=Config.ENABLE_BRIDGE)
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False,
    )