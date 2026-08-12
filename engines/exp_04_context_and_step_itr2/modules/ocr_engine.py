import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..system_instruction import OCR_SYSTEM_INSTRUCTION, OCR_USER_PROMPT


OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))


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
        return {
            "type": "input_image",
            "image_url": source,
        }

    if source.startswith(("http://", "https://")):
        return {
            "type": "input_image",
            "image_url": source,
        }

    image_path = Path(source)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg"

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded_image}",
    }


def get_json_ocr(source: str):
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
            print(f"[ocr_engine.get_json_ocr] OCR completed with {len(cleaned_steps)} steps")
            return cleaned_steps

    print("[ocr_engine.get_json_ocr] OCR returned no valid steps")
    return []
