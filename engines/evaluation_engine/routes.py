import json
import logging

from flask import Blueprint, Response, jsonify, request

from .modules.analysis_engine import analyze_solution
from question_analysis import analyze_question_skill_weightage_with_meta


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("evaluation_engine_api", __name__)
# Preserve the original import name for existing application bootstraps.
api_routes = api_blueprint


@api_blueprint.route("/", methods=["GET"])
def health_check_endpoint():
    return "context and step evaluation service is running"


@api_blueprint.route("/get_json_ocr", methods=["POST"])
def extract_ocr_endpoint():
    from .modules.ocr_engine import extract_ocr_steps

    data = request.get_json(silent=True) or {}
    source = data.get("source")
    logger.info("OCR request received for source: %s", source)

    if not source:
        return jsonify({"error": "source is required"}), 400

    try:
        result = extract_ocr_steps(source)
        return jsonify({"ocr_data": result}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/checked_json_ocr", methods=["POST"])
def evaluate_ocr_endpoint():
    data = request.get_json(silent=True) or {}
    ocr_data = data.get("ocr_data")
    solution_source = data.get("solution_url") or data.get("source")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    full_marks = data.get("full_marks")
    logger.info("OCR evaluation request received")

    if (not isinstance(ocr_data, list) or not ocr_data) and not solution_source:
        return jsonify({"error": "Either ocr_data or solution_url is required"}), 400

    try:
        result = analyze_solution(
            image_source=solution_source or "",
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            ocr_data=ocr_data if isinstance(ocr_data, list) and ocr_data else None,
            full_marks=full_marks,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/checked_json_ocr_with_rag", methods=["POST"])
def evaluate_ocr_with_rag_endpoint():
    from .modules.ocr_engine import extract_ocr_steps
    from .modules.testing_engine import evaluate_ocr_steps_with_rag

    data = request.get_json(silent=True) or {}
    ocr_data = data.get("ocr_data")
    solution_source = data.get("solution_url") or data.get("source")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    full_marks = data.get("full_marks")
    logger.info("RAG-backed OCR evaluation request received")

    if (not isinstance(ocr_data, list) or not ocr_data) and not solution_source:
        return jsonify({"error": "Either ocr_data or solution_url is required"}), 400

    try:
        if not isinstance(ocr_data, list) or not ocr_data:
            ocr_data = extract_ocr_steps(solution_source)

        result = evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            full_marks=full_marks,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/index_documents", methods=["POST"])
def index_documents_endpoint():
    from .modules.testing_engine import index_documents

    data = request.get_json(silent=True) or {}
    documents = data.get("documents")
    collection_name = data.get("collection_name")

    if not isinstance(documents, list) or not documents:
        return jsonify({"error": "documents must be a non-empty list"}), 400

    try:
        if collection_name:
            result = index_documents(
                documents=documents,
                collection_name=collection_name,
            )
        else:
            result = index_documents(documents=documents)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/index_text_documents", methods=["POST"])
def index_text_documents_endpoint():
    from .modules.testing_engine import index_text_documents

    data = request.get_json(silent=True) or {}
    document_paths = data.get("document_paths")
    collection_name = data.get("collection_name")

    if not isinstance(document_paths, list) or not document_paths:
        return jsonify({"error": "document_paths must be a non-empty list"}), 400

    try:
        if collection_name:
            result = index_text_documents(
                document_paths=document_paths,
                collection_name=collection_name,
            )
        else:
            result = index_text_documents(document_paths=document_paths)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/evaluated_json_ocr", methods=["POST"])
def evaluation_status_endpoint():
    return "json_ocr evaluated successfully"


@api_blueprint.route("/api/v1/question-analysis/analyze", methods=["POST"])
def analyze_question_endpoint():
    data = request.get_json(silent=True)

    try:
        result = analyze_question_skill_weightage_with_meta(data)
        return Response(
            response=json.dumps(result["weights"]),
            status=200,
            mimetype="application/json",
            headers={"X-Question-Analysis-Mode": result["mode"]},
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_blueprint.route("/get_analysis", methods=["POST"])
def analyze_solution_endpoint():
    data = request.get_json(silent=True) or {}
    image_source = data.get("image_source", "")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    full_marks = data.get("full_marks")
    logger.info("Solution analysis request received for image source: %s", image_source)

    if not image_source:
        return jsonify({"error": "image_source is required"}), 400

    try:
        result = analyze_solution(
            image_source=image_source,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            full_marks=full_marks,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
