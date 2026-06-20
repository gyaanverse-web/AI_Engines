import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import httpx


GEMINI_OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))


def _get_gemini_sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Gemini SDK not installed. Install dependencies from "
            "engines/exp_01_gemini/requirements.txt"
        ) from exc

    return genai, types


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
                "stepId": str(step.get("stepId", "")),
                "text": _strip_text_wrappers(step.get("text", "")),
            }
        )
    return cleaned_steps


def _load_image_bytes(source: str) -> tuple[bytes, str]:
    if source.startswith("data:image/"):
        header, encoded = source.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
        return base64.b64decode(encoded), mime_type

    if source.startswith(("http://", "https://")):
        response = httpx.get(source, timeout=60.0)
        response.raise_for_status()
        mime_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        return response.content, mime_type

    image_path = Path(source)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    return image_path.read_bytes(), mime_type or "image/jpeg"


def get_json_ocr(source: str):
    genai, types = _get_gemini_sdk()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    image_bytes, mime_type = _load_image_bytes(source)
    schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stepId": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["stepId", "text"],
                },
            }
        },
        "required": ["steps"],
    }

    response = client.models.generate_content(
        model=GEMINI_OCR_MODEL,
        contents=[
            "Return strict JSON with a top-level key 'steps'. "
            "Each item must contain 'stepId' as a string and 'text' as a string. "
            "The 'text' value must be an OCR-faithful LaTeX-style transcription of that line. "
            "Convert the whole line into LaTeX-friendly notation so it can be analyzed by an AI step by step. "
            "Keep wording faithful to the image even if it contains mistakes. If text is unclear, keep the "
            "uncertain OCR as written instead of correcting it.",
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            system_instruction=(
                "Perform OCR transcription only. Do not solve, interpret, normalize, simplify, "
                "or correct any student writing. Preserve the original wording, symbols, order, "
                "mistakes, and line-by-line structure as closely as possible. Return every line in "
                "LaTeX-friendly form. Use standard LaTeX for mathematical expressions, symbols, fractions, "
                "powers, roots, and equations. Keep ordinary words as plain text whenever possible "
                "instead of wrapping them in \\text{...}. Represent units with \\mathrm{...}. Do not use Markdown dollar "
                "delimiters like $...$ or $$...$$. Do not add explanations, common understanding, missing "
                "steps, or corrections. Do not rewrite incorrect statements into correct ones. Do not use "
                "Markdown code fences."
            ),
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    parsed_output = json.loads(response.text or "{}")
    if isinstance(parsed_output, dict):
        steps = parsed_output.get("steps", [])
        if isinstance(steps, list):
            cleaned_steps = _clean_ocr_steps(steps)
            print(f"[gemini.ocr_engine.get_json_ocr] OCR completed with {len(cleaned_steps)} steps")
            return cleaned_steps

    print("[gemini.ocr_engine.get_json_ocr] OCR returned no valid steps")
    return []
