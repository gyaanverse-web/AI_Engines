import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask


def load_environment() -> None:
    package_dir = Path(__file__).resolve().parent
    load_dotenv(package_dir.parent / ".env")
    load_dotenv(package_dir / ".env")


def create_app(url_prefix: str = "/evaluation_engine") -> Flask:
    load_environment()

    from .routes import api_blueprint

    app = Flask(__name__)
    log_level_name = os.getenv("EVALUATION_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    app.logger.setLevel(log_level)
    for handler in app.logger.handlers:
        handler.setLevel(log_level)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("EVALUATION_MAX_REQUEST_BYTES", str(20 * 1024 * 1024))
    )
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)
    return app
