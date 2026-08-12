from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

from exp_04_context_and_step_itr2 import create_app


app = create_app(url_prefix="/context_and_step_itr2")

if __name__ == "__main__":
    app.run(debug=True)
