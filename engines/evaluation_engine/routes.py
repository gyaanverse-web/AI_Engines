import logging

from flask import Blueprint, jsonify, request

from .modules.analysis_engine import analyze_solution


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("evaluation_engine_api", __name__)
# Preserve the original import name for existing application bootstraps.
api_routes = api_blueprint
MAX_OCR_STEPS = 250


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _optional_string(data: dict, key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _top_k(data: dict) -> int:
    value = data.get("top_k", 5)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    return value


def _full_marks(data: dict) -> float | None:
    value = data.get("full_marks")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("full_marks must be a positive number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("full_marks must be a positive number") from exc
    if numeric <= 0 or numeric > 10000:
        raise ValueError("full_marks must be greater than 0 and at most 10000")
    return numeric


def _ocr_data(data: dict) -> list[dict] | None:
    value = data.get("ocr_data")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("ocr_data must be an array")
    if len(value) > MAX_OCR_STEPS:
        raise ValueError(f"ocr_data must contain at most {MAX_OCR_STEPS} steps")
    for index, step in enumerate(value, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"ocr_data step {index} must be an object")
        if not isinstance(step.get("text"), str):
            raise ValueError(f"ocr_data step {index} text must be a string")
    return value or None


def _source_log_label(source: str) -> str:
    if source.startswith("data:"):
        return "data-url"
    if source.startswith(("http://", "https://")):
        return "remote-url"
    return "local-path"


def _processing_error(message: str, exc: Exception):
    logger.exception("%s: %s", message, exc)
    return jsonify({"error": message}), 500


@api_blueprint.route("/", methods=["GET"])
def health_check_endpoint():
    return "evaluation engine service is running"


@api_blueprint.route("/get_json_ocr", methods=["POST"])
def extract_ocr_endpoint():
    from .modules.ocr_engine import extract_ocr_steps

    try:
        data = _json_body()
        source = _optional_string(data, "source")
        if not source:
            raise ValueError("source is required")
        logger.info("OCR request received (%s)", _source_log_label(source))
        result = extract_ocr_steps(source)
        return jsonify({"ocr_data": result}), 200
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("OCR processing failed", exc)


@api_blueprint.route("/checked_json_ocr", methods=["POST"])
def evaluate_ocr_endpoint():
    try:
        data = _json_body()
        ocr_data = _ocr_data(data)
        solution_source = _optional_string(data, "solution_url") or _optional_string(data, "source")
        question = _optional_string(data, "question")
        full_marks = _full_marks(data)
        logger.info("OCR evaluation request received")
        if ocr_data is None and not solution_source:
            raise ValueError("Either ocr_data or solution_url is required")
        result = analyze_solution(
            image_source=solution_source or "",
            question=question,
            ocr_data=ocr_data,
            full_marks=full_marks,
            use_rag=False,
        )
        return jsonify(result), 200
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("Solution evaluation failed", exc)


@api_blueprint.route("/checked_json_ocr_with_rag", methods=["POST"])
def evaluate_ocr_with_rag_endpoint():
    from .modules.ocr_engine import extract_ocr_steps
    from .modules.testing_engine import evaluate_ocr_steps_with_rag

    try:
        data = _json_body()
        ocr_data = _ocr_data(data)
        solution_source = _optional_string(data, "solution_url") or _optional_string(data, "source")
        question = _optional_string(data, "question")
        collection_name = _optional_string(data, "collection_name") or None
        top_k = _top_k(data)
        full_marks = _full_marks(data)
        logger.info("RAG-backed OCR evaluation request received")
        if ocr_data is None and not solution_source:
            raise ValueError("Either ocr_data or solution_url is required")
        if ocr_data is None:
            ocr_data = extract_ocr_steps(solution_source)

        result = evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            full_marks=full_marks,
        )
        return jsonify(result), 200
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("RAG-backed solution evaluation failed", exc)


@api_blueprint.route("/index_documents", methods=["POST"])
def index_documents_endpoint():
    from .modules.testing_engine import index_documents

    try:
        data = _json_body()
        documents = data.get("documents")
        collection_name = _optional_string(data, "collection_name") or None
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty array")
        if len(documents) > 100:
            raise ValueError("documents must contain at most 100 items")
        if collection_name:
            result = index_documents(
                documents=documents,
                collection_name=collection_name,
            )
        else:
            result = index_documents(documents=documents)
        return jsonify(result), 200
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("Document indexing failed", exc)


@api_blueprint.route("/index_text_documents", methods=["POST"])
def index_text_documents_endpoint():
    from .modules.testing_engine import index_text_documents

    try:
        data = _json_body()
        document_paths = data.get("document_paths")
        collection_name = _optional_string(data, "collection_name") or None
        if not isinstance(document_paths, list) or not document_paths:
            raise ValueError("document_paths must be a non-empty array")
        if len(document_paths) > 100 or not all(isinstance(path, str) and path.strip() for path in document_paths):
            raise ValueError("document_paths must contain 1 to 100 non-empty strings")
        if collection_name:
            result = index_text_documents(
                document_paths=document_paths,
                collection_name=collection_name,
            )
        else:
            result = index_text_documents(document_paths=document_paths)
        return jsonify(result), 200
    except (TypeError, ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("Text document indexing failed", exc)


@api_blueprint.route("/evaluated_json_ocr", methods=["POST"])
def evaluation_status_endpoint():
    return "json_ocr evaluated successfully"


@api_blueprint.route("/get_analysis", methods=["POST"])
def analyze_solution_endpoint():
    try:
        data = _json_body()
        image_source = _optional_string(data, "image_source")
        question = _optional_string(data, "question")
        collection_name = _optional_string(data, "collection_name") or None
        top_k = _top_k(data)
        full_marks = _full_marks(data)
        if not image_source:
            raise ValueError("image_source is required")
        logger.info("Solution analysis request received (%s)", _source_log_label(image_source))
        result = analyze_solution(
            image_source=image_source,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            full_marks=full_marks,
            use_rag=True,
        )
        return jsonify(result), 200
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return _processing_error("Solution analysis failed", exc)
