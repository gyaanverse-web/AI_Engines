from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

load_dotenv(Path(__file__).resolve().parent / ".env")

from routes import api_routes


app = Flask(__name__)
app.register_blueprint(api_routes)

if __name__ == "__main__":
    app.run(debug=True)
