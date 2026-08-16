import json
import logging

from flask import Blueprint, Response, jsonify, request

from . import analyze_question_skill_weightage_with_meta


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("question_analysis_api", __name__)


@api_blueprint.route("/", methods=["GET"])
def health_check_endpoint():
    return "question analysis service is running"


@api_blueprint.route("/analyze", methods=["POST"])
def analyze_question_endpoint():
    payload = request.get_json(silent=True)
    logger.info("Question analysis request received")

    try:
        result = analyze_question_skill_weightage_with_meta(payload)
        return Response(
            response=json.dumps(result["weights"]),
            status=200,
            mimetype="application/json",
            headers={"X-Question-Analysis-Mode": result["mode"]},
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Question analysis failed")
        return jsonify({"error": str(exc)}), 500
