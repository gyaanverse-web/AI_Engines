from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

load_dotenv(Path(__file__).resolve().parent / ".env")

from exp_01_gemini.routes import api_routes as gemini_routes
from exp_02_openai.routes import api_routes as openai_routes
from exp_03_openai_ocr_gemini_test.routes import api_routes as openai_ocr_gemini_test_routes
from exp_04_context_and_step_itr2.routes import api_routes as context_and_step_itr2_routes


app = Flask(__name__)
app.register_blueprint(openai_routes, url_prefix="/openai")
app.register_blueprint(gemini_routes, url_prefix="/gemini")
app.register_blueprint(
    openai_ocr_gemini_test_routes,
    url_prefix="/openai_ocr_gemini_test",
)
app.register_blueprint(
    context_and_step_itr2_routes,
    url_prefix="/context_and_step_itr2",
)

if __name__ == "__main__":
    app.run(debug=True)
