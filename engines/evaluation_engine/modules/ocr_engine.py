import base64
import binascii
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..system_instruction import OCR_SYSTEM_INSTRUCTION, OCR_USER_PROMPT


OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OCR_MAX_IMAGE_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
ENGINE_ROOT = Path(__file__).resolve().parents[2]
_ocr_local_file_root = Path(os.getenv("OCR_LOCAL_FILE_ROOT", "..")).expanduser()
OCR_LOCAL_FILE_ROOT = (
    _ocr_local_file_root
    if _ocr_local_file_root.is_absolute()
    else ENGINE_ROOT / _ocr_local_file_root
).resolve()
OCR_ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
logger = logging.getLogger(__name__)


def _strip_text_wrappers(text: str) -> str:
    cleaned = text or ""
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", cleaned)
    return " ".join(cleaned.split()).strip()


def _clean_ocr_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_steps = []
    for step in steps:
        cleaned_steps.append(
            {
                "stepId": step.get("stepId", ""),
                "text": _strip_text_wrappers(step.get("text", "")),
            }
        )
    return cleaned_steps


def _build_image_input(source: str) -> dict[str, Any]:
    if source.startswith("data:image/"):
        header, separator, encoded = source.partition(",")
        mime_type = header.removeprefix("data:").split(";", 1)[0].lower()
        if not separator or ";base64" not in header.lower():
            raise ValueError("Image data URL must contain base64-encoded data")
        if mime_type not in OCR_ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {mime_type}")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image data URL contains invalid base64 data") from exc
        if not image_bytes:
            raise ValueError("Image data URL is empty")
        if len(image_bytes) > OCR_MAX_IMAGE_BYTES:
            raise ValueError(f"Image must not exceed {OCR_MAX_IMAGE_BYTES} bytes")
        return {
            "type": "input_image",
            "image_url": source,
        }

    if source.startswith(("http://", "https://")):
        return {
            "type": "input_image",
            "image_url": source,
        }

    image_path = Path(source).expanduser()
    if not image_path.is_absolute():
        image_path = OCR_LOCAL_FILE_ROOT / image_path
    image_path = image_path.resolve()
    try:
        image_path.relative_to(OCR_LOCAL_FILE_ROOT)
    except ValueError as exc:
        raise PermissionError(
            f"Image path must be inside OCR_LOCAL_FILE_ROOT: {OCR_LOCAL_FILE_ROOT}"
        ) from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {source}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in OCR_ALLOWED_MIME_TYPES:
        raise ValueError("Local image must be JPEG, PNG, WEBP, or GIF")
    if image_path.stat().st_size > OCR_MAX_IMAGE_BYTES:
        raise ValueError(f"Image must not exceed {OCR_MAX_IMAGE_BYTES} bytes")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded_image}",
    }


def extract_ocr_steps(source: str) -> list[dict[str, Any]]:
    """Extract line-preserving OCR steps from an image source."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model=OPENAI_OCR_MODEL,
        temperature=OPENAI_TEMPERATURE,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": OCR_SYSTEM_INSTRUCTION,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": OCR_USER_PROMPT,
                    },
                    _build_image_input(source),
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ocr_steps",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "stepId": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["stepId", "text"],
                            },
                        }
                    },
                    "required": ["steps"],
                },
            }
        },
    )

    parsed_output = json.loads(response.output_text)
    if isinstance(parsed_output, dict):
        steps = parsed_output.get("steps", [])
        if isinstance(steps, list):
            cleaned_steps = _clean_ocr_steps(steps)
            logger.info("OCR completed with %s steps", len(cleaned_steps))
            return cleaned_steps

    logger.warning("OCR returned no valid steps")
    return []


# Backward-compatible alias for existing integrations.
get_json_ocr = extract_ocr_steps
