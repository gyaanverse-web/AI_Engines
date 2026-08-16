import ast
import json
import logging
import math
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..system_instruction import (
    DOCUMENT_RAG_SYSTEM_INSTRUCTION,
    NUMERIC_VERIFICATION_SYSTEM_INSTRUCTION,
    build_document_rag_prompt,
    build_numeric_verification_prompt,
)


GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
NVIDIA_EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nv-embedqa-e5-v5",
)
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000"))
RAG_MAX_DOCUMENT_BYTES = int(os.getenv("RAG_MAX_DOCUMENT_BYTES", str(5 * 1024 * 1024)))
ENGINE_ROOT = Path(__file__).resolve().parents[2]
_rag_document_root = Path(os.getenv("RAG_DOCUMENT_ROOT", "../Data")).expanduser()
RAG_DOCUMENT_ROOT = (
    _rag_document_root
    if _rag_document_root.is_absolute()
    else ENGINE_ROOT / _rag_document_root
).resolve()
QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME_EXP_04",
    os.getenv("QDRANT_COLLECTION_NAME", "evaluation_engine"),
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_VECTOR_SIZE = int(os.getenv("EMBEDDING_VECTOR_SIZE", "4096"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "50"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
NUMERIC_TOLERANCE = float(os.getenv("NUMERIC_TOLERANCE", "0.05"))
logger = logging.getLogger(__name__)

STANDARD_FORMULAS = {
    "kinematics_displacement": {
        "topic": "Kinematics",
        "canonical": r"S = ut + \frac{1}{2}at^2",
    }
}


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def _get_nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
if QDRANT_URL == "http://localhost:6333":
    qdrant_client = QdrantClient(path="qdrant_local_db")
else:
    qdrant_client() -> QdrantClient:
    return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        check_compatibility=False,
    )


def _get_gemini_sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Gemini SDK not installed. Install dependencies from "
            "engines/evaluation_engine/requirements.txt"
        ) from exc

    return genai, types


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        collected = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(part_text)
        if collected:
            return "".join(collected)

    return ""


def _generate_json_response(
    *,
    model: str,
    system_instruction: str,
    user_content: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    genai, types = _get_gemini_sdk()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(_extract_response_text(response) or "{}")


def _generate_text_response(
    *,
    model: str,
    system_instruction: str,
    user_content: Any,
) -> str:
    genai, types = _get_gemini_sdk()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            system_instruction=system_instruction,
        ),
    )
    return _extract_response_text(response    )


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalize_latex_text(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return cleaned

    replacements = {
        "∴": r"\therefore",
        "×": r"\times",
        "÷": r"\div",
        "−": "-",
        "√": r"\sqrt{}",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", cleaned)

    cleaned = re.sub(r"\$\$(.*?)\$\$", r"\1", cleaned)
    cleaned = re.sub(r"\$(.*?)\$", r"\1", cleaned)
    cleaned = re.sub(
        r"(\d)(m/s(?:\^2|²)|m\\,s\^\{-2\}|m/s|kg|cm|mm|km|N|m|s)\b",
        r"\1 \\mathrm{\2}",
        cleaned,
    )
    return cleaned


def _strip_latex_commands(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\left|\\right|\\therefore", "", cleaned)
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned


def _compact_math(text: str) -> str:
    return re.sub(r"\s+", "", _strip_latex_commands(text or "")).lower()


def _replace_frac(expr: str) -> str:
    pattern = re.compile(r"\\frac\(([^()]+)\)\(([^()]+)\)")
    previous = None
    current = expr
    while previous != current:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def _latex_expr_to_python(expr: str) -> str:
    cleaned = _strip_latex_commands(expr)
    cleaned = cleaned.replace("\\times", "*").replace("\\div", "/")
    cleaned = cleaned.replace("^", "**")
    cleaned = _replace_frac(cleaned)
    cleaned = re.sub(r"\bAns\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", cleaned)
    cleaned = re.sub(r"(?<=[)])(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\()", "*", cleaned)
    return cleaned


def _safe_eval_numeric(expr: str, variable_values: dict[str, float] | None = None) -> float | None:
    python_expr = _latex_expr_to_python(expr)
    if variable_values:
        for name, value in variable_values.items():
            python_expr = re.sub(rf"\b{re.escape(name)}\b", str(value), python_expr)

    if len(python_expr) > 256 or re.search(r"[^0-9a-zA-Z_+\-*/().]", python_expr):
        return None

    try:
        parsed = ast.parse(python_expr, mode="eval")
        value = _evaluate_numeric_ast(parsed.body)
        if not math.isfinite(value) or abs(value) > 1e100:
            return None
        return float(value)
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return None


def _evaluate_numeric_ast(node: ast.AST) -> float:
    """Evaluate a deliberately small arithmetic-only AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_numeric_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if not isinstance(node, ast.BinOp):
        raise ValueError("unsupported numeric expression")

    left = _evaluate_numeric_ast(node.left)
    right = _evaluate_numeric_ast(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        if abs(right) > 12 or abs(left) > 1e12:
            raise ValueError("unsafe exponent")
        return left ** right
    raise ValueError("unsupported numeric operator")


def _extract_equation(text: str) -> tuple[str, str] | None:
    if "=" not in text:
        return None
    left, right = text.split("=", 1)
    left = left.strip()
    # Keep balanced parentheses in expressions such as 2(11x + 5).  Only
    # discard punctuation or unmatched OCR closing parentheses at the end.
    right = re.split(r"[.!?;\n]\s*(?=[A-Za-z]|$)", right, maxsplit=1)[0].strip()
    while right.endswith(")") and right.count(")") > right.count("("):
        right = right[:-1].rstrip()
    if not left or not right:
        return None
    return left, right


def _extract_assigned_variable(step_text: str) -> str | None:
    match = re.search(r"(?:^|\\therefore\s*)([A-Za-z])\s*=", step_text)
    if match:
        return match.group(1)
    return None


def _extract_assigned_numeric_value(step_text: str, variable: str) -> float | None:
    """Extract an assigned numeric expression, including fractions such as ``-4/5``."""
    equation = _extract_equation(step_text)
    if not equation:
        return None

    left, right = equation
    assigned = _extract_assigned_variable(f"{left}=")
    if assigned != variable:
        return None

    right = re.sub(r"\\mathrm\{[^}]*\}", "", right)
    right = re.sub(
        r"\s+(?:m/s(?:\^2|²)|m\\,s\^\{-2\}|m/s|kg|cm|mm|km|N|newtons?|m|s)\s*$",
        "",
        right,
        flags=re.IGNORECASE,
    )
    return _safe_eval_numeric(right)


def _extract_last_number(text: str) -> float | None:
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", _strip_latex_commands(text))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _extract_last_number_text(text: str) -> str | None:
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", _strip_latex_commands(text))
    if not numbers:
        return None
    return numbers[-1]


def _decimal_places(number_text: str | None) -> int | None:
    if not number_text or "." not in number_text:
        return 0 if number_text else None
    return len(number_text.split(".", 1)[1])


def _canonicalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    compact = re.sub(r"\s+", "", unit).replace("²", "^2").lower()
    aliases = {
        "n": "N",
        "newton": "N",
        "newtons": "N",
        "m/s^2": "m/s^2",
        "m/s2": "m/s^2",
        "m\\,s^{-2}": "m/s^2",
        "ms^{-2}": "m/s^2",
        "m/s": "m/s",
        "kg": "kg",
        "cm": "cm",
        "mm": "mm",
        "km": "km",
        "m": "m",
        "s": "s",
    }
    return aliases.get(compact, unit.strip())


def _extract_unit(text: str) -> str | None:
    match = re.search(r"\\mathrm\{([^}]*)\}", text)
    if match:
        return _canonicalize_unit(match.group(1))

    plain_match = re.search(
        r"(?<![A-Za-z])(m/s(?:\^2|²)|m/s|kg|cm|mm|km|N|newtons?|m|s)\b",
        _strip_latex_commands(text),
        flags=re.IGNORECASE,
    )
    if plain_match:
        return _canonicalize_unit(plain_match.group(1))
    return None


def _infer_expected_unit(question: str, local_context: str, step_text: str) -> str | None:
    step_plain = _strip_latex_commands(step_text)
    variable = _extract_assigned_variable(step_text)
    step_lower = step_plain.lower()

    local_rules = [
        ("N", r"\b(?:force|friction|tension|thrust|weight)\b", {"f"}),
        ("m/s^2", r"\bacceleration\b", {"a"}),
        ("m/s", r"\b(?:velocity|speed)\b", {"u", "v"}),
        ("kg", r"\bmass\b", {"m"}),
        ("m", r"\b(?:distance|displacement|length|height)\b", {"s", "d", "h"}),
        ("s", r"\btime\b", {"t"}),
    ]
    for unit, pattern, _ in local_rules:
        if re.search(pattern, step_lower):
            return unit

    nearby = f"{local_context}".lower()
    nearby_units = {unit for unit, pattern, _ in local_rules if re.search(pattern, nearby)}
    if len(nearby_units) == 1:
        return nearby_units.pop()

    question_lower = question.lower()
    question_units = {unit for unit, pattern, _ in local_rules if re.search(pattern, question_lower)}
    contextual_units = nearby_units | question_units
    for unit, _, variables in local_rules:
        if variable and variable.lower() in variables and unit in contextual_units:
            return unit
    return question_units.pop() if len(question_units) == 1 else None


def _solve_linear_value_from_equation(equation_text: str, variable: str) -> float | None:
    equation = _extract_equation(equation_text)
    if not equation:
        return None

    left_expr, right_expr = equation
    left_at_zero = _safe_eval_numeric(left_expr, {variable: 0})
    left_at_one = _safe_eval_numeric(left_expr, {variable: 1})
    right_at_zero = _safe_eval_numeric(right_expr, {variable: 0})
    right_at_one = _safe_eval_numeric(right_expr, {variable: 1})

    values = [left_at_zero, left_at_one, right_at_zero, right_at_one]
    if any(value is None for value in values):
        return None

    left_constant = left_at_zero
    left_coeff = left_at_one - left_at_zero
    right_constant = right_at_zero
    right_coeff = right_at_one - right_at_zero
    denominator = left_coeff - right_coeff
    if abs(denominator) < 1e-9:
        return None

    return (right_constant - left_constant) / denominator


def _find_expected_numeric_value(
    variable: str,
    local_context: str,
) -> float | None:
    lines = []
    for raw_line in (local_context or "").splitlines():
        _, separator, content = raw_line.partition(":")
        if not separator:
            content = raw_line
        cleaned = content.strip()
        if variable in cleaned and "=" in cleaned:
            lines.append(cleaned)

    for line in reversed(lines):
        value = _solve_linear_value_from_equation(line, variable)
        if value is not None:
            return value
    return None


def _format_number(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        rounded = round(value, decimals)
        if decimals == 0:
            return str(int(round(rounded)))
        return f"{rounded:.{decimals}f}"

    rounded = round(value, 4)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _make_result(
    status: str,
    description: str,
    *,
    step_weight: float | None = None,
) -> dict[str, Any]:
    result = {
        "step_status": status,
        "description": description,
    }
    if step_weight is not None:
        result["step_weight"] = step_weight
    return result


def _apply_rule_override(
    step_result: dict[str, Any],
    rule_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not rule_result:
        return step_result

    step_result["step_status"] = rule_result["step_status"]
    step_result["description"] = rule_result["description"]
    if "step_weight" in rule_result:
        step_result["step_weight"] = rule_result["step_weight"]
    return _align_step_result(step_result)


def _is_heading_or_setup_line(step_text: str) -> bool:
    normalized = _compact_math(step_text)
    plain = _normalize_whitespace(_strip_latex_commands(step_text)).lower()
    setup_phrases = [
        "given that",
        "we know that",
        "find",
        "determine",
        "final acceleration",
    ]
    if any(phrase in plain for phrase in setup_phrases):
        return True
    if plain.endswith("?"):
        return True
    return normalized in {"", "ans:-giventhat", "weknowthat"}


def _step_type_check(step_text: str) -> dict[str, Any] | None:
    plain = _normalize_whitespace(_strip_latex_commands(step_text))
    lowered = plain.lower()
    if _is_heading_or_setup_line(step_text):
        if "final acceleration" in lowered or lowered.endswith("?"):
            return _make_result(
                "incomplete",
                "Missing: solution step; Correct step: start solving for acceleration",
                step_weight=0.1,
            )
        if "given that" in lowered or "we know that" in lowered:
            return _make_result(
                "incomplete",
                "Missing: actual setup values; Correct step: list the known quantities",
                step_weight=0.1,
            )
        return _make_result(
            "incomplete",
            "Missing: mathematical step; Correct step: continue the solution",
            step_weight=0.1,
        )
    return None


def _formula_checker(step_text: str, question: str, local_context: str) -> dict[str, Any] | None:
    normalized = _compact_math(step_text)
    combined = f"{question} {local_context} {step_text}".lower()

    if "acceleration" in combined and "s=ut+" in normalized and "at^2" in normalized:
        has_half = any(token in normalized for token in [r"\frac(1)(2)", "1/2", "0.5"])
        if not has_half:
            return _make_result(
                "wrong",
                r"Error: the displacement formula misses the \frac{1}{2} factor on at^2; "
                r"Correct step: use S = ut + \frac{1}{2}at^2",
                step_weight=0.7,
            )
        return _make_result(
            "right",
            "The displacement formula is correct",
            step_weight=0.7,
        )

    return None


def _apply_deterministic_step_checks(
    step_result: dict[str, Any],
    question: str,
    step_text: str,
    local_context: str,
) -> dict[str, Any]:
    variable = _extract_assigned_variable(step_text)
    reported_value = (
        _extract_assigned_numeric_value(step_text, variable)
        if variable
        else _extract_last_number(step_text)
    )
    expected_unit = _infer_expected_unit(question, local_context, step_text)
    reported_unit = _extract_unit(step_text)

    if variable and reported_value is not None:
        expected_value = _find_expected_numeric_value(
            variable,
            f"{question}\n{local_context}",
        )
        if expected_value is not None:
            if abs(reported_value - expected_value) <= NUMERIC_TOLERANCE:
                if expected_unit and not reported_unit:
                    step_result["step_status"] = "incomplete"
                    step_result["description"] = (
                        f"Missing: the unit {expected_unit} is not written; "
                        f"Correct step: write \\therefore {variable} = "
                        f"{_format_number(expected_value)}\\ \\mathrm{{{expected_unit}}}"
                    )
                elif expected_unit and reported_unit and reported_unit != expected_unit:
                    step_result["step_status"] = "wrong"
                    step_result["description"] = (
                        f"Error: the numerical value is correct, but the unit {reported_unit} "
                        f"is incorrect; Correct step: write \\therefore {variable} = "
                        f"{_format_number(expected_value)}\\ \\mathrm{{{expected_unit}}}"
                    )
                else:
                    step_result["step_status"] = "right"
                    step_result["description"] = ""
            else:
                step_result["step_status"] = "wrong"
                # Never round the correct answer to the student's precision.  For
                # example, a wrong `x = -1` must not turn the correct `x = -0.8`
                # into another `x = -1` in the feedback.
                correction = _format_number(expected_value)
                if expected_unit:
                    step_result["description"] = (
                        f"Error: the final calculation gives the wrong value for {variable}; "
                        f"Correct step: write \\therefore {variable} = {correction}"
                        f"\\ \\mathrm{{{expected_unit}}}"
                    )
                else:
                    step_result["description"] = (
                        f"Error: the final calculation gives the wrong value for {variable}; "
                        f"Correct step: write \\therefore {variable} = {correction}"
                    )

    return _align_step_result(step_result)


def _shorten_description(description: str) -> str:
    cleaned = _normalize_whitespace(description)
    if not cleaned:
        return cleaned

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    first_sentence = parts[0].strip()
    first_sentence = first_sentence.rstrip(" .,!?:;")
    if len(first_sentence) <= 120:
        return first_sentence

    truncated = first_sentence[:120]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(" .,!?:;")


def _align_step_result(step_result: dict[str, Any]) -> dict[str, Any]:
    status = str(step_result.get("step_status", "")).strip().lower()
    description = _shorten_description(str(step_result.get("description", "")))

    if status == "right":
        step_result["description"] = ""
        step_result["step_status"] = status
        return step_result

    if status in {"incomplete", "unknown"}:
        step_result["description"] = description
        step_result["step_status"] = status
        return step_result

    if not description:
        step_result["description"] = description
        return step_result

    lowered = description.lower()
    if status == "wrong" and any(
        phrase in lowered
        for phrase in ["numeric value is correct", "is correct but", "correct but the unit is missing"]
    ):
        status = "incomplete"
        description = ""
    elif status == "wrong" and not any(
        token in lowered
        for token in ["should", "correct", "instead", "use", "must be", "replace", "missing"]
    ):
        corrected_step = step_result.get("text", "").strip()
        if corrected_step:
            description = f"Error: {description}; Correct step: write {corrected_step}"
        else:
            description = f"Error: {description}; Correct step: write the corrected mathematical step"

    step_result["step_status"] = status
    step_result["description"] = description
    return step_result


def _build_local_step_context(
    ocr_data: list[dict[str, Any]],
    current_index: int,
    window: int = 2,
) -> str:
    nearby_steps = []
    start = max(0, current_index - window)
    end = min(len(ocr_data), current_index + window + 1)

    for index in range(start, end):
        if index == current_index:
            continue
        step = ocr_data[index]
        step_id = step.get("stepId", str(index + 1))
        step_text = _normalize_latex_text(step.get("text", ""))
        if step_text:
            nearby_steps.append(f"{step_id}: {step_text}")

    return "\n".join(nearby_steps)


def _needs_numeric_verification(step_text: str) -> bool:
    normalized = step_text.lower()
    has_number = bool(re.search(r"\d", step_text))
    answer_hint = any(token in normalized for token in ["ans", "therefore", r"\therefore", "="])
    return has_number and answer_hint


def _verify_numeric_step(
    step_result: dict[str, Any],
    question: str,
    step_text: str,
    local_context: str,
) -> dict[str, Any]:
    verification = _generate_json_response(
        model=GEMINI_CHAT_MODEL,
        system_instruction=NUMERIC_VERIFICATION_SYSTEM_INSTRUCTION,
        user_content=build_numeric_verification_prompt(
            question=_normalize_latex_text(question or "Not provided"),
            step_text=step_text,
            local_context=local_context,
        ),
        schema={
            "type": "object",
            "properties": {
                "step_status": {
                    "type": "string",
                    "enum": ["right", "wrong", "unknown", "incomplete"],
                },
                "description": {"type": "string"},
            },
            "required": ["step_status", "description"],
        },
    )
    step_result["step_status"] = verification["step_status"]
    step_result["description"] = verification["description"]
    return _align_step_result(step_result)


def _normalize_ocr_steps(ocr_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized_steps = []
    for index, step in enumerate(ocr_data, start=1):
        step_id = str(step.get("stepId", index))
        step_text = _normalize_latex_text(step.get("text", "").strip())
        if not step_text:
            continue
        normalized_steps.append({"stepId": step_id, "text": step_text})
    return normalized_steps


def _post_process_step_result(
    step_result: dict[str, Any],
    ocr_data: list[dict[str, str]],
    question: str,
    index: int,
) -> dict[str, Any]:
    step_text = _normalize_latex_text(step_result.get("text", "").strip())
    step_result["text"] = step_text
    local_context = _build_local_step_context(ocr_data, index)

    step_result = _align_step_result(step_result)
    step_result = _apply_rule_override(step_result, _step_type_check(step_text))
    step_result = _apply_rule_override(
        step_result,
        _formula_checker(step_text, question, local_context),
    )
    step_result = _apply_deterministic_step_checks(
        step_result=step_result,
        question=question,
        step_text=step_text,
        local_context=local_context,
    )
    return step_result


def _coerce_step_results(
    ocr_data: list[dict[str, str]],
    raw_results: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    raw_by_step_id = {
        str(item.get("stepId", "")): item
        for item in raw_results
        if isinstance(item, dict)
    }
    final_results = []

    for index, original_step in enumerate(ocr_data):
        step_id = original_step["stepId"]
        raw_step = raw_by_step_id.get(step_id, {})
        step_result = {
            "stepId": step_id,
            "text": raw_step.get("text", original_step["text"]),
            "step_status": raw_step.get("step_status", "unknown"),
            "step_weight": raw_step.get("step_weight", 0.5),
            "topic": _normalize_whitespace(str(raw_step.get("topic", ""))),
            "step_understanding": _normalize_whitespace(
                str(raw_step.get("step_understanding", ""))
            ),
            "description": str(raw_step.get("description", "")),
        }
        final_results.append(
            _post_process_step_result(
                step_result=step_result,
                ocr_data=ocr_data,
                question=question,
                index=index,
            )
        )

    return final_results


def create_qdrant_collection(collection_name: str = QDRANT_COLLECTION_NAME):
    collection_name = _validate_collection_name(collection_name)
    qdrant_client = _get_qdrant_client()
    if qdrant_client.collection_exists(collection_name=collection_name):
        return collection_name

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    return collection_name


def chunk_document_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks = []
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        start += chunk_size - overlap

    return chunks


def get_embedding(text: str):
    response = _get_nvidia_client().embeddings.create(
        model=NVIDIA_EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
        extra_body={"input_type": "passage"}
    )
    return response.data[0].embedding


def get_embeddings(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE):
    if not texts:
        return []

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = _get_nvidia_client().embeddings.create(
            model=NVIDIA_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": "passage"}
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def index_documents(
    documents: list[dict[str, Any]],
    collection_name: str = QDRANT_COLLECTION_NAME,
):
    collection_name = _validate_collection_name(collection_name)
    create_qdrant_collection(collection_name)

    chunk_records = []
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise TypeError(f"document {index} must be an object")
        document_id = str(document.get("document_id") or uuid.uuid4()).strip()
        document_text = document.get("text", "")
        metadata = document.get("metadata", {})
        if not isinstance(document_text, str):
            raise TypeError(f"document {index} text must be a string")
        if not isinstance(metadata, dict):
            raise TypeError(f"document {index} metadata must be an object")

        for chunk_index, chunk_text in enumerate(chunk_document_text(document_text)):
            chunk_records.append(
                {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

    points = []
    if chunk_records:
        embeddings = get_embeddings([record["text"] for record in chunk_records])
        if len(embeddings) != len(chunk_records):
            raise RuntimeError("embedding provider returned an unexpected number of vectors")
        for record, embedding in zip(chunk_records, embeddings):
            if len(embedding) != EMBEDDING_VECTOR_SIZE:
                raise ValueError(
                    "embedding vector size does not match EMBEDDING_VECTOR_SIZE "
                    f"({len(embedding)} != {EMBEDDING_VECTOR_SIZE})"
                )
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{collection_name}:{record['document_id']}:{record['chunk_index']}",
                        )
                    ),
                    vector=embedding,
                    payload=record,
                )
            )

        for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
            batch = points[start:start + QDRANT_UPSERT_BATCH_SIZE]
            _get_qdrant_client().upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

    return {"collection_name": collection_name, "indexed_chunks": len(points)}


def _resolve_rag_document_path(file_path: str) -> Path:
    text_path = Path(file_path).expanduser()
    if not text_path.is_absolute():
        text_path = RAG_DOCUMENT_ROOT.parent / text_path
    text_path = text_path.resolve()
    try:
        text_path.relative_to(RAG_DOCUMENT_ROOT)
    except ValueError as exc:
        raise PermissionError(
            f"Document path must be inside RAG_DOCUMENT_ROOT: {RAG_DOCUMENT_ROOT}"
        ) from exc
    if not text_path.is_file():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    if text_path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("RAG text documents must use a .txt or .md extension")
    if text_path.stat().st_size > RAG_MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"RAG text documents must not exceed {RAG_MAX_DOCUMENT_BYTES} bytes"
        )
    return text_path


def extract_text_file_content(file_path: str):
    return _resolve_rag_document_path(file_path).read_text(encoding="utf-8").strip()


def index_text_documents(
    document_paths: list[str],
    collection_name: str = QDRANT_COLLECTION_NAME,
):
    documents = []
    logger.info("Starting text indexing for %s documents", len(document_paths))

    for document_path in document_paths:
        logger.info("Indexing document: %s", document_path)
        text_path = _resolve_rag_document_path(document_path)
        document_text = extract_text_file_content(str(text_path))
        if not document_text:
            continue

        documents.append({
            "document_id": text_path.stem,
            "text": document_text,
            "metadata": {
                "source_path": str(text_path),
                "file_name": text_path.name,
            },
        })

    return index_documents(
        documents=documents,
        collection_name=collection_name,
    )


def retrieve_relevant_chunks(
    query: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
):
    query = str(query or "").strip()
    if not query:
        raise ValueError("retrieval query must not be empty")
    collection_name = _validate_collection_name(collection_name)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")

    query_vector = get_embedding(query)
    search_result = _get_qdrant_client().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": point.score,
            "document_id": point.payload.get("document_id"),
            "chunk_index": point.payload.get("chunk_index"),
            "text": point.payload.get("text", ""),
            "metadata": point.payload.get("metadata", {}),
        }
        for point in search_result.points
    ]


def _validate_collection_name(collection_name: str | None) -> str:
    cleaned = str(collection_name or "").strip()
    if not cleaned:
        cleaned = QDRANT_COLLECTION_NAME
    if len(cleaned) > 128 or not re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned):
        raise ValueError(
            "collection_name must contain only letters, numbers, underscores, dots, or hyphens"
        )
    return cleaned


def generate_document_answer(
    question: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
):
    relevant_chunks = retrieve_relevant_chunks(
        query=question,
        collection_name=collection_name,
        top_k=top_k,
    )

    context = "\n\n".join(
        f"[Chunk {chunk['chunk_index']} | Score {chunk['score']:.4f}]\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    return {
        "question": question,
        "answer": _generate_text_response(
            model=GEMINI_CHAT_MODEL,
            system_instruction=DOCUMENT_RAG_SYSTEM_INSTRUCTION,
            user_content=build_document_rag_prompt(
                question=question,
                context=context,
            ),
        ),
        "sources": relevant_chunks,
    }


def evaluate_ocr_steps(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    full_marks: float | None = None,
):
    """Compatibility entry point for the canonical flat evaluator."""
    from .testing_engine import evaluate_ocr_steps as evaluate_steps

    return evaluate_steps(
        ocr_data=ocr_data,
        question=question,
        full_marks=full_marks,
    )


def evaluate_ocr_steps_with_rag(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
    full_marks: float | None = None,
):
    """Compatibility entry point for the canonical flat RAG evaluator."""
    from .testing_engine import evaluate_ocr_steps_with_rag as evaluate_steps_with_rag

    return evaluate_steps_with_rag(
        ocr_data=ocr_data,
        question=question,
        collection_name=collection_name,
        top_k=top_k,
        full_marks=full_marks,
    )
