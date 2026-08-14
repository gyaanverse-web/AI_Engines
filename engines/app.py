import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify

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


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe. Deliberately makes no upstream calls.

    Two consumers, and they want opposite things from a health check:

      * the platform (Railway) restarts the service when this stops answering,
      * the backend's circuit breaker uses cheap failures to decide the engine
        is down without paying EVAL_ENGINE_TIMEOUT_MS per discovery.

    Both are served by an endpoint that answers instantly and reflects only
    *this process*. Calling OpenAI or Gemini from here would make a provider
    hiccup look like a dead container and get the service restarted mid-grade —
    and it would hand the breaker a 120s hang exactly when it is trying to
    avoid one.

    `deps` reports whether the SDKs a grading request needs are importable. The
    live incident on 2026-08-12 was a missing `google-genai`, where OCR
    succeeded, the evaluate call failed, and the failure was only visible after
    an OCR call had already been paid for. That is a boot-time fact, so it is
    reported here instead of being rediscovered per request.
    """
    return jsonify(
        {
            "status": "ok",
            "deps": {
                "openai": _importable("openai"),
                "google_genai": _importable("google.genai"),
                "qdrant_client": _importable("qdrant_client"),
            },
            "keys": {
                "openai": bool(os.environ.get("OPENAI_API_KEY")),
                "gemini": bool(
                    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                ),
            },
        }
    )


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


if __name__ == "__main__":
    # Local convenience only. The real server is gunicorn — see railway.json:
    #
    #   gunicorn app:app --worker-class gthread --workers 2 --threads 8 --timeout 180
    #
    # Flask's dev server used to run here with debug=True, which is
    # single-threaded: every OCR and evaluate call is a multi-second blocking
    # request to OpenAI/Gemini, so one in-flight grade blocked every other
    # request on the box. That made a local throughput measurement meaningless
    # and would have been a hard ceiling anywhere it reached production.
    # threaded=True at least overlaps those waits, which is enough for
    # development. gunicorn does not run on Windows, so a throughput number
    # measured here is not the number production will produce — measure that
    # against a deployed engine, not this.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true"},
        threaded=True,
    )
