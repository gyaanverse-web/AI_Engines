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
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)
    return app
