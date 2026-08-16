import os

from flask import Flask

from .routes import api_blueprint


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(api_blueprint, url_prefix="/question_analysis")
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"})
