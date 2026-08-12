import os

from evaluation_engine import create_app


app = create_app(url_prefix="/evaluation_engine")


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"})
