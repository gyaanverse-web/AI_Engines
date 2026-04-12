# import os
# from mpxpy.mathpix_client import MathpixClient



# def get_final_ocr_data(ocr_data):
#     final_ocr_data = []
#     for index, line in enumerate(ocr_data, start=1):
#         line_text = line.get("text", "")
#         if line_text.strip():
#             final_ocr_data.append({
#                 "stepId": line.get("id", index),
#                 "text": line_text,
#             })
#     return final_ocr_data
    
    
    
    
# def get_json_ocr(source: str):
#     client = MathpixClient(
#         app_id=os.getenv("MATHPIX_APP_ID"),
#         app_key=os.getenv("MATHPIX_APP_KEY"),
#         api_url=os.getenv("MATHPIX_URL"),
#     )

#     if source.startswith(("http://", "https://")):
#         image = client.image_new(url=source, include_line_data=True, include_detected_alphabets=True)
#     else:
#         image = client.image_new(file_path=source, include_line_data=True)
        
#     ocr_data = image.lines_json()
#     final_ocr_data = get_final_ocr_data(ocr_data)

#     return final_ocr_data

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


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
    print(f"[ocr_engine.get_json_ocr] Starting OCR for source: {source}")
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
                        "text": (
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
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Return strict JSON with a top-level key 'steps'. "
                            "Each item must contain 'stepId' as a string and 'text' as a string. "
                            "The 'text' value must be an OCR-faithful LaTeX-style transcription of that line. "
                            "Convert the whole line into LaTeX-friendly notation so it can be analyzed by an AI step by step. "
                            "Keep wording faithful to the image even if it contains mistakes. If text is unclear, keep the "
                            "uncertain OCR as written instead of correcting it."
                        ),
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
