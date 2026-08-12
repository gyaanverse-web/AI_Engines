import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

from evaluation_engine import create_app as create_evaluation_app
from question_analysis.routes import api_blueprint as question_analysis_blueprint


app = create_evaluation_app(url_prefix="/evaluation_engine")
app.register_blueprint(question_analysis_blueprint, url_prefix="/question_analysis")

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"})
