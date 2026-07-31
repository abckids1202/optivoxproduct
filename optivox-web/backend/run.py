"""Development entry point for the OptiVox backend."""
from app.config import Config
from app.main import create_app


if __name__ == "__main__":
    app = create_app(start_bridge=Config.ENABLE_BRIDGE)
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "8000")),
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False,
    )
