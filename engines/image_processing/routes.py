import logging

from flask import Blueprint, jsonify, request


logger = logging.getLogger(__name__)
api_blueprint = Blueprint("image_processing_api", __name__)


def _json_image_source() -> str:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object containing image_source")
    image_source = data.get("image_source")
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError("image_source must be a non-empty string")
    return image_source.strip()


@api_blueprint.route("/", methods=["GET"])
def health_check_endpoint():
    return "image processing service is running"


@api_blueprint.route("/contains_text", methods=["POST"])
def contains_text_endpoint():
    from .modules.text_detector import detect_text

    try:
        image_source = _json_image_source()
        logger.info("Text-presence request received (image_source)")
        result = detect_text(image_source)

        return jsonify({"contains_text": result}), 200
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Image text detection failed: %s", exc)
        return jsonify({"error": "Image text detection failed"}), 500
