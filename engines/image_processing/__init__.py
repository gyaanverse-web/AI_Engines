import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge


def load_environment() -> None:
    package_dir = Path(__file__).resolve().parent
    load_dotenv(package_dir.parent / ".env")
    load_dotenv(package_dir / ".env")


def create_app(url_prefix: str = "/image_processing") -> Flask:
    load_environment()

    from .routes import api_blueprint

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("IMAGE_PROCESSING_MAX_REQUEST_BYTES", str(12 * 1024 * 1024))
    )
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error: RequestEntityTooLarge):
        return jsonify({"error": "Image upload is too large"}), 413

    return app


__all__ = ["create_app", "load_environment"]
