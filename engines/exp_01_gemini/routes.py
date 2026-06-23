from flask import Blueprint, jsonify, request

from .modules.analysis_engine import analyzer
from .modules.ocr_engine import get_json_ocr as run_get_json_ocr
from .modules.testing_engine import (
    evaluate_ocr_steps_with_rag as run_evaluate_ocr_steps_with_rag,
    index_documents as run_index_documents,
    index_text_documents as run_index_text_documents,
)


api_routes = Blueprint("gemini_routes", __name__)


@api_routes.route("/", methods=["GET"])
def welcome():
    return "welcome to flask engine server"


@api_routes.route("/get_json_ocr", methods=["POST"])
def get_json_ocr():
    data = request.get_json(silent=True) or {}
    source = data.get("source")
    print(f"[routes.get_json_ocr] Request received for source: {source}")

    if not source:
        return jsonify({"error": "source is required"}), 400

    try:
        result = run_get_json_ocr(source)
        return jsonify({"ocr_data": result}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_routes.route("/checked_json_ocr", methods=["POST"])
def checked_json_ocr():
    data = request.get_json(silent=True) or {}
    ocr_data = data.get("ocr_data")
    solution_url = data.get("solution_url") or data.get("source")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    print("[routes.checked_json_ocr] Request received")

    if (not isinstance(ocr_data, list) or not ocr_data) and not solution_url:
        return (
            jsonify({"error": "Either ocr_data or solution_url is required"}),
            400,
        )

    try:
        result = analyzer(
            image_source=solution_url or "",
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            ocr_data=ocr_data if isinstance(ocr_data, list) and ocr_data else None,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_routes.route("/checked_json_ocr_with_rag", methods=["POST"])
def checked_json_ocr_with_rag():
    data = request.get_json(silent=True) or {}
    ocr_data = data.get("ocr_data")
    solution_url = data.get("solution_url") or data.get("source")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    print("[routes.checked_json_ocr_with_rag] Request received")

    if (not isinstance(ocr_data, list) or not ocr_data) and not solution_url:
        return (
            jsonify({"error": "Either ocr_data or solution_url is required"}),
            400,
        )

    try:
        if not isinstance(ocr_data, list) or not ocr_data:
            ocr_data = run_get_json_ocr(solution_url)

        result = run_evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_routes.route("/index_documents", methods=["POST"])
def index_documents():
    data = request.get_json(silent=True) or {}
    documents = data.get("documents")
    collection_name = data.get("collection_name")

    if not isinstance(documents, list) or not documents:
        return jsonify({"error": "documents must be a non-empty list"}), 400

    try:
        if collection_name:
            result = run_index_documents(
                documents=documents,
                collection_name=collection_name,
            )
        else:
            result = run_index_documents(documents=documents)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_routes.route("/index_text_documents", methods=["POST"])
def index_text_documents():
    data = request.get_json(silent=True) or {}
    document_paths = data.get("document_paths")
    collection_name = data.get("collection_name")

    if not isinstance(document_paths, list) or not document_paths:
        return jsonify({"error": "document_paths must be a non-empty list"}), 400

    try:
        if collection_name:
            result = run_index_text_documents(
                document_paths=document_paths,
                collection_name=collection_name,
            )
        else:
            result = run_index_text_documents(
                document_paths=document_paths,
            )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_routes.route("/evaluated_json_ocr", methods=["POST"])
def evaluated_json_ocr():
    return "json_ocr evaluated successfully"


@api_routes.route("/get_analysis", methods=["POST"])
def get_analysis():
    data = request.get_json(silent=True) or {}
    image_source = data.get("image_source", "")
    question = data.get("question", "")
    collection_name = data.get("collection_name")
    top_k = data.get("top_k", 5)
    print(f"[routes.get_analysis] Request received for image_source: {image_source}")

    if not image_source:
        return jsonify({"error": "image_source is required"}), 400

    try:
        result = analyzer(image_source=image_source, question=question, collection_name=collection_name, top_k=top_k)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
