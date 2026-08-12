import logging
import re
from typing import Any

from question_analysis.categories import CATEGORY_KEYS
from question_analysis.normalizer import normalize_scores
from question_analysis.preprocessor import normalize_text
from question_analysis.scorer import calculate_rule_scores
from ..system_instruction import (
    RECONSTRUCT_SOLUTION_SYSTEM_INSTRUCTION,
    SOLUTION_PROFILE_SYSTEM_INSTRUCTION,
    STEP_EVALUATION_SYSTEM_INSTRUCTION,
    build_reconstruct_solution_prompt,
    build_solution_profile_prompt,
    build_step_evaluation_prompt,
)
from .provider_engine import (
    GEMINI_CHAT_MODEL,
    QDRANT_COLLECTION_NAME,
    RAG_MIN_SCORE,
    _apply_deterministic_step_checks,
    _decimal_places,
    _extract_assigned_variable,
    _extract_last_number,
    _extract_last_number_text,
    _format_number,
    _formula_checker,
    _generate_json_response,
    _normalize_latex_text,
    index_documents,
    index_text_documents,
    retrieve_relevant_chunks,
)


STATUS_VALUES = {"right", "wrong", "unknown", "incomplete"}
STATUS_CREDITS = {
    "right": 1.0,
    "incomplete": 0.5,
    "unknown": 0.25,
    "wrong": 0.0,
}
STEP_TYPE_DEFAULT = "concept_based"
STEP_TYPE_CORRECTIONS = {
    "calculation_based": "continue the calculation carefully",
    "formula_based": "write the correct formula or substitution",
    "concept_based": "state the correct concept briefly",
    "memory_based": "write the correct fact or value",
    "proof_or_derivation_based": "continue the derivation logically",
    "reasoning_based": "justify the conclusion from the previous step",
    "language_or_explanation_based": "complete the explanation briefly",
    "data_interpretation_based": "read the data and state the inference",
    "diagram_based": "draw or label the required diagram step",
    "application_based": "apply the concept to the given situation",
}
QUESTION_PART_PATTERN = re.compile(r"\(([A-Za-z0-9]+)\)\s*")
STEP_PART_PATTERN = re.compile(r"^\s*(?:\(([A-Za-z0-9]+)\)|([A-Za-z0-9]+)\))\s*")
logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalize_ocr_steps(ocr_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized_steps = []
    for index, step in enumerate(ocr_data, start=1):
        step_id = str(step.get("stepId", index))
        step_text = _normalize_latex_text(str(step.get("text", "")).strip())
        if not step_text:
            continue
        normalized_steps.append({"stepId": step_id, "text": step_text})
    return normalized_steps


def _is_numbering_artifact(text: str) -> bool:
    cleaned = text.strip()
    return bool(re.fullmatch(r"[\[(]?[1-9][\])]?\.?", cleaned))


def _reconstruct_solution_steps(
    ocr_steps: list[dict[str, str]],
    question: str,
) -> list[dict[str, Any]]:
    if not ocr_steps:
        return []

    filtered_steps = [step for step in ocr_steps if not _is_numbering_artifact(step["text"])]
    if not filtered_steps:
        filtered_steps = ocr_steps

    parsed_response = _generate_json_response(
        model=GEMINI_CHAT_MODEL,
        system_instruction=RECONSTRUCT_SOLUTION_SYSTEM_INSTRUCTION,
        user_content=build_reconstruct_solution_prompt(
            question=question,
            ocr_steps=filtered_steps,
        ),
        schema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stepId": {"type": "string"},
                            "sourceStepIds": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "text": {"type": "string"},
                        },
                        "required": ["stepId", "sourceStepIds", "text"],
                    },
                }
            },
            "required": ["steps"],
        },
    )

    reconstructed_steps = parsed_response.get("steps", []) if isinstance(parsed_response, dict) else []
    final_steps = []

    for index, step in enumerate(reconstructed_steps, start=1):
        source_step_ids = [
            str(step_id).strip()
            for step_id in step.get("sourceStepIds", [])
            if str(step_id).strip()
        ]
        text = _normalize_latex_text(str(step.get("text", "")).strip())
        if not text:
            continue
        final_steps.append(
            {
                "stepId": str(step.get("stepId", index)),
                "sourceStepIds": source_step_ids or [str(step.get("stepId", index))],
                "text": text,
            }
        )

    if final_steps:
        return final_steps

    return [
        {
            "stepId": step["stepId"],
            "sourceStepIds": [step["stepId"]],
            "text": step["text"],
        }
        for step in filtered_steps
    ]


def _extract_question_parts(question: str) -> list[dict[str, str]]:
    normalized_question = _normalize_whitespace(question)
    matches = list(QUESTION_PART_PATTERN.finditer(normalized_question))
    if len(matches) < 2:
        return []

    parts = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_question)
        label = str(match.group(1)).strip().lower()
        text = normalized_question[start:end].strip(" ,;:.")
        if index + 1 < len(matches):
            text = re.sub(r"(?:,\s*)?(?:and|or)\s*$", "", text, flags=re.IGNORECASE).strip(" ,;:.")
        if text:
            parts.append({"label": label, "text": text})
    return parts


def _extract_part_label_from_text(text: str) -> str | None:
    match = STEP_PART_PATTERN.match(text or "")
    if not match:
        return None

    label = match.group(1) or match.group(2)
    cleaned_label = str(label).strip().lower()
    return cleaned_label or None


def _build_blocks_from_steps(
    steps: list[dict[str, Any]],
    question_parts: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not steps:
        return [], []

    label_to_part = {
        str(part["label"]).strip().lower(): _normalize_whitespace(str(part["text"]))
        for part in question_parts
        if str(part.get("label", "")).strip()
    }
    annotated_steps: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None

    for index, step in enumerate(steps, start=1):
        detected_label = _extract_part_label_from_text(step["text"])
        if current_block is None or detected_label is not None:
            block_id = f"block_{len(blocks) + 1}"
            current_block = {
                "blockId": block_id,
                "blockLabel": detected_label or block_id,
                "question_part_label": "",
                "question_part_text": "",
                "stepIds": [],
            }
            if detected_label and detected_label in label_to_part:
                current_block["question_part_label"] = detected_label
                current_block["question_part_text"] = label_to_part[detected_label]
            blocks.append(current_block)

        current_block["stepIds"].append(step["stepId"])
        annotated_steps.append(
            {
                **step,
                "blockId": current_block["blockId"],
                "blockLabel": current_block["blockLabel"],
                "question_part_label": current_block["question_part_label"],
                "question_part_text": current_block["question_part_text"],
            }
        )

    if question_parts and len(blocks) == len(question_parts):
        for block, question_part in zip(blocks, question_parts):
            if not block["question_part_label"]:
                block["question_part_label"] = question_part["label"]
                block["question_part_text"] = question_part["text"]
                block["blockLabel"] = question_part["label"]

    block_meta_by_id = {block["blockId"]: block for block in blocks}
    hydrated_steps = []
    for step in annotated_steps:
        block_meta = block_meta_by_id[step["blockId"]]
        hydrated_steps.append(
            {
                **step,
                "blockLabel": block_meta["blockLabel"],
                "question_part_label": block_meta["question_part_label"],
                "question_part_text": block_meta["question_part_text"],
            }
        )

    return hydrated_steps, list(block_meta_by_id.values())


def _serialize_blocks_for_prompt(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for step in steps:
        block_id = str(step.get("blockId", "")).strip() or "block_1"
        if block_id not in blocks_by_id:
            ordered_ids.append(block_id)
            blocks_by_id[block_id] = {
                "blockId": block_id,
                "blockLabel": str(step.get("blockLabel", "")).strip() or block_id,
                "question_part_label": str(step.get("question_part_label", "")).strip(),
                "question_part_text": str(step.get("question_part_text", "")).strip(),
                "steps": [],
            }
        blocks_by_id[block_id]["steps"].append(
            {
                "stepId": step["stepId"],
                "sourceStepIds": step.get("sourceStepIds", []),
                "text": step["text"],
            }
        )

    return [blocks_by_id[block_id] for block_id in ordered_ids]


def _build_solution_context(steps: list[dict[str, Any]]) -> str:
    return "\n".join(f"{step['stepId']}: {step['text']}" for step in steps)


def _build_retrieval_query(question: str, steps: list[dict[str, Any]]) -> str:
    solution_context = _build_solution_context(steps)
    if question:
        return f"Question:\n{question}\n\nStudent solution:\n{solution_context}"
    return f"Student solution:\n{solution_context}"


def _get_rag_context(
    question: str,
    steps: list[dict[str, Any]],
    collection_name: str,
    top_k: int,
) -> tuple[str, str | None]:
    try:
        relevant_chunks = retrieve_relevant_chunks(
            query=_build_retrieval_query(question, steps),
            collection_name=collection_name,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "", f"retrieval_error: {exc}"

    if not relevant_chunks:
        return "", "no_context"

    grounded_chunks = [
        chunk
        for chunk in relevant_chunks
        if isinstance(chunk.get("score"), (int, float)) and chunk["score"] >= RAG_MIN_SCORE
    ]
    if not grounded_chunks:
        return "", "context_issue"

    context = "\n\n".join(chunk.get("text", "") for chunk in grounded_chunks if chunk.get("text"))
    return context, None if context else "context_issue"


def _pick_primary_category(
    scores: dict[str, float],
    tie_breaker: dict[str, int] | None = None,
) -> str:
    tie_breaker = tie_breaker or {}
    return max(
        CATEGORY_KEYS,
        key=lambda category: (
            float(scores.get(category, 0.0)),
            int(tie_breaker.get(category, 0)),
            -CATEGORY_KEYS.index(category),
        ),
    )


def _build_question_profile(question: str) -> dict[str, Any]:
    normalized_question = normalize_text(question)
    scores = calculate_rule_scores(question=normalized_question, chapter="")
    weights = normalize_scores(scores)
    ordered_categories = sorted(
        CATEGORY_KEYS,
        key=lambda category: (weights.get(category, 0), -CATEGORY_KEYS.index(category)),
        reverse=True,
    )
    return {
        "scores": scores,
        "weights": weights,
        "primary_category": ordered_categories[0],
        "secondary_categories": ordered_categories[1:3],
    }


def _build_step_type_scores(
    *,
    question_profile: dict[str, Any],
    question: str,
    step_text: str,
    step_understanding: str,
    topic: str,
) -> dict[str, float]:
    scores = {
        category: float(question_profile["scores"].get(category, 0.0)) * 0.25
        for category in CATEGORY_KEYS
    }
    combined_text = normalize_text(
        " ".join(part for part in [question, step_text, step_understanding, topic] if part)
    )
    step_scores = calculate_rule_scores(question=combined_text, chapter="")
    for category in CATEGORY_KEYS:
        scores[category] += float(step_scores.get(category, 0.0))

    if re.search(r"\d", step_text):
        scores["calculation_based"] += 12
    if any(token in step_text for token in ["=", "+", "-", "*", "/", "^", r"\frac", r"\times", r"\div"]):
        scores["calculation_based"] += 10
        scores["formula_based"] += 8
    if re.search(r"\bformula|substitute|using\b", combined_text):
        scores["formula_based"] += 14
    if re.search(r"\btherefore|hence|thus|so\b", combined_text):
        scores["reasoning_based"] += 8
    if re.search(r"\bprove|derive|assume|let\b", combined_text):
        scores["proof_or_derivation_based"] += 18
        scores["reasoning_based"] += 8
    if re.search(r"\bexplain|describe|because|reason|why\b", combined_text):
        scores["concept_based"] += 12
        scores["language_or_explanation_based"] += 8
    if re.search(r"\bdefine|state|list|given|known\b", combined_text):
        scores["memory_based"] += 10
    if re.search(r"\btable|chart|graph|data|observe\b", combined_text):
        scores["data_interpretation_based"] += 18
    if re.search(r"\bdiagram|figure|draw|label|circuit|map\b", combined_text):
        scores["diagram_based"] += 18
    if re.search(r"\breal life|daily life|practical|application|case study\b", combined_text):
        scores["application_based"] += 18

    word_count = len(combined_text.split())
    has_sentence_flow = word_count >= 10 and not any(token in step_text for token in ["=", r"\frac"])
    if has_sentence_flow:
        scores["language_or_explanation_based"] += 8
        scores["concept_based"] += 6

    return scores


def _resolve_step_type(
    *,
    raw_step_type: str,
    question_profile: dict[str, Any],
    question: str,
    step_text: str,
    step_understanding: str,
    topic: str,
) -> str:
    step_scores = _build_step_type_scores(
        question_profile=question_profile,
        question=question,
        step_text=step_text,
        step_understanding=step_understanding,
        topic=topic,
    )
    predicted_step_type = _pick_primary_category(step_scores, question_profile["weights"])
    raw_step_type = str(raw_step_type).strip()

    if raw_step_type in CATEGORY_KEYS:
        raw_score = float(step_scores.get(raw_step_type, 0.0))
        best_score = float(step_scores.get(predicted_step_type, 0.0))
        if raw_score >= max(15.0, best_score * 0.75):
            return raw_step_type

    return predicted_step_type or STEP_TYPE_DEFAULT


def _build_solution_profile(
    *,
    question: str,
    steps: list[dict[str, Any]],
    question_profile: dict[str, Any],
    context: str,
    question_parts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not steps:
        return {
            "goal": "",
            "method_outline": [],
            "carry_forward_policy": "",
            "global_notes": [],
            "primary_category": question_profile["primary_category"],
        }

    parsed_response = _generate_json_response(
        model=GEMINI_CHAT_MODEL,
        system_instruction=SOLUTION_PROFILE_SYSTEM_INSTRUCTION,
        user_content=build_solution_profile_prompt(
            question=question,
            question_parts=question_parts or [],
            question_profile=question_profile,
            solution_blocks=_serialize_blocks_for_prompt(steps),
            reconstructed_steps=steps,
            retrieved_context=context,
        ),
        schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "method_outline": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "carry_forward_policy": {"type": "string"},
                "global_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "primary_category": {
                    "type": "string",
                    "enum": CATEGORY_KEYS,
                },
            },
            "required": [
                "goal",
                "method_outline",
                "carry_forward_policy",
                "global_notes",
                "primary_category",
            ],
        },
    )

    if isinstance(parsed_response, dict):
        return parsed_response

    return {
        "goal": "",
        "method_outline": [],
        "carry_forward_policy": "",
        "global_notes": [],
        "primary_category": question_profile["primary_category"],
    }


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "..."


def _coerce_description(status: str, description: str, step_type: str) -> str:
    cleaned = " ".join((description or "").split()).strip()
    correction_hint = STEP_TYPE_CORRECTIONS.get(step_type, STEP_TYPE_CORRECTIONS[STEP_TYPE_DEFAULT])

    if status == "right":
        return ""

    if status == "unknown":
        return cleaned or "Step is unclear from OCR or missing context."

    if status == "incomplete":
        if not cleaned:
            cleaned = f"Missing: the next logical detail; Correct step: {correction_hint}"
        elif not cleaned.startswith("Missing:"):
            cleaned = f"Missing: {cleaned}"
        if "; Correct step:" not in cleaned:
            cleaned = f"{cleaned.rstrip('. ')}; Correct step: {correction_hint}"
        missing_part, _, correction = cleaned.partition("; Correct step:")
        return f"{missing_part.strip()}; Correct step: {correction.strip()}"

    if not cleaned:
        cleaned = f"Error: the step is not mathematically valid; Correct step: {correction_hint}"
    elif not cleaned.startswith("Error:"):
        cleaned = f"Error: {cleaned}"
    if "; Correct step:" not in cleaned:
        cleaned = f"{cleaned.rstrip('. ')}; Correct step: {correction_hint}"
    error_part, _, correction = cleaned.partition("; Correct step:")
    return f"{error_part.strip()}; Correct step: {correction.strip()}"


def _clamp_step_weight(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _coerce_step_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in STATUS_VALUES else "unknown"


def _coerce_topic(topic: Any, step_type: str) -> str:
    cleaned = " ".join(str(topic or "").split()).strip()
    if cleaned:
        return _truncate_words(cleaned, 6)
    return step_type.replace("_based", "").replace("_", " ").title()


def _coerce_step_understanding(value: Any) -> str:
    return _normalize_whitespace(
        str(value or "Step intent is not fully clear.")
    )


def _strip_latex_commands(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\left|\\right|\\therefore", "", cleaned)
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned


def _is_heading_like_step(step_text: str) -> bool:
    plain = _normalize_whitespace(_strip_latex_commands(step_text)).strip()
    if not plain:
        return True

    plain_without_label = STEP_PART_PATTERN.sub("", plain, count=1).strip()
    if not plain_without_label:
        return True

    if "=" in plain_without_label:
        return False

    if step_text.strip().endswith(":") and len(plain_without_label.split()) <= 8:
        return True

    return bool(re.fullmatch(r"[A-Za-z0-9]+", plain_without_label))


def _build_local_step_context(
    steps: list[dict[str, Any]],
    current_index: int,
    window: int = 2,
) -> str:
    nearby_steps = []
    start = max(0, current_index - window)
    end = min(len(steps), current_index + window + 1)

    for index in range(start, end):
        if index == current_index:
            continue
        step = steps[index]
        step_text = _normalize_latex_text(str(step.get("text", "")).strip())
        if step_text:
            nearby_steps.append(f"{step['stepId']}: {step_text}")

    return "\n".join(nearby_steps)


def _apply_rule_override(
    step_result: dict[str, Any],
    rule_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not rule_result:
        return step_result

    step_result["step_status"] = _coerce_step_status(rule_result.get("step_status", step_result["step_status"]))
    step_result["description"] = str(rule_result.get("description", step_result["description"])).strip()
    if "step_weight" in rule_result:
        step_result["step_weight"] = _clamp_step_weight(rule_result["step_weight"])
    return step_result


def _extract_reported_unit(text: str) -> str | None:
    latex_match = re.search(r"\\mathrm\{([^}]*)\}", text or "")
    if latex_match:
        return latex_match.group(1).strip()

    plain_match = re.search(
        r"\b(m/s\^2|m/s|kg|cm|mm|km|m|s|N|newton|newtons)\b",
        _strip_latex_commands(text or ""),
        flags=re.IGNORECASE,
    )
    if not plain_match:
        return None

    unit = plain_match.group(1).strip()
    if unit.lower() in {"newton", "newtons", "n"}:
        return "N"
    return unit


def _infer_expected_unit_from_context(
    question: str,
    local_context: str,
    step_text: str,
) -> str | None:
    combined = f"{question} {local_context} {step_text}".lower()
    if "acceleration" in combined:
        return "m/s^2"
    if "velocity" in combined or "speed" in combined:
        return "m/s"
    if "distance" in combined or "displacement" in combined:
        return "m"
    if "mass" in combined:
        return "kg"
    if "force" in combined or "friction" in combined or "pull" in combined or "tension" in combined:
        return "N"
    if "time" in combined:
        return "s"
    return None


def _apply_context_unit_guard(
    *,
    step_result: dict[str, Any],
    question: str,
    step_text: str,
    local_context: str,
) -> dict[str, Any]:
    if step_result["step_status"] != "right":
        return step_result

    variable = _extract_assigned_variable(step_text)
    reported_value = _extract_last_number(step_text)
    reported_value_text = _extract_last_number_text(step_text)
    decimals = _decimal_places(reported_value_text)
    expected_unit = _infer_expected_unit_from_context(question, local_context, step_text)
    reported_unit = _extract_reported_unit(step_text)

    if not variable or reported_value is None or not expected_unit:
        return step_result

    formatted_value = _format_number(reported_value, decimals)
    if not reported_unit:
        step_result["step_status"] = "incomplete"
        step_result["description"] = (
            f"Missing: the unit {expected_unit} is not written; "
            f"Correct step: write \\therefore {variable} = {formatted_value}\\ \\mathrm{{{expected_unit}}}"
        )
        return step_result

    if reported_unit != expected_unit:
        step_result["step_status"] = "wrong"
        step_result["description"] = (
            f"Error: the numerical value is correct, but the unit {reported_unit} is incorrect; "
            f"Correct step: write \\therefore {variable} = {formatted_value}\\ \\mathrm{{{expected_unit}}}"
        )
    return step_result


def _post_process_step_result(
    *,
    question: str,
    steps: list[dict[str, Any]],
    index: int,
    step_result: dict[str, Any],
) -> dict[str, Any]:
    step_text = _normalize_latex_text(step_result["text"])
    local_context = _build_local_step_context(steps, index)

    if _is_heading_like_step(step_text):
        step_result["step_status"] = "right"
        step_result["description"] = ""
        step_result["step_weight"] = min(step_result["step_weight"], 0.05)
        return step_result

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
    step_result = _apply_context_unit_guard(
        step_result=step_result,
        question=question,
        step_text=step_text,
        local_context=local_context,
    )
    step_result["step_status"] = _coerce_step_status(step_result["step_status"])
    step_result["step_weight"] = _clamp_step_weight(step_result["step_weight"])
    return step_result


def _round_score(value: float) -> float:
    return round(float(value), 2)


def _clean_marks_value(value: float) -> float | int:
    rounded = _round_score(value)
    if float(rounded).is_integer():
        return int(rounded)
    return rounded


def _coerce_full_marks(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _status_credit(status: str) -> float:
    return STATUS_CREDITS.get(status, STATUS_CREDITS["unknown"])


def _aggregate_status(statuses: list[str]) -> str:
    if any(status == "wrong" for status in statuses):
        return "wrong"
    if any(status == "incomplete" for status in statuses):
        return "incomplete"
    if any(status == "unknown" for status in statuses):
        return "unknown"
    return "right"


def _summarize_status_counts(step_results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "right": sum(1 for item in step_results if item["step_status"] == "right"),
        "wrong": sum(1 for item in step_results if item["step_status"] == "wrong"),
        "unknown": sum(1 for item in step_results if item["step_status"] == "unknown"),
        "incomplete": sum(1 for item in step_results if item["step_status"] == "incomplete"),
    }


def _resolve_step_weights(step_results: list[dict[str, Any]]) -> dict[str, float]:
    resolved = {
        item["stepId"]: max(0.0, float(item.get("step_weight", 0.0)))
        for item in step_results
    }
    if sum(resolved.values()) > 0:
        return resolved
    return {item["stepId"]: 1.0 for item in step_results}


def _build_block_results(
    step_results: list[dict[str, Any]],
    full_marks: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not step_results:
        summary = {
            "overall_status": "unknown",
            "status_breakdown": {key: 0 for key in sorted(STATUS_VALUES)},
            "step_count": 0,
            "block_count": 0,
        }
        if full_marks is not None:
            summary["full_marks"] = _clean_marks_value(full_marks)
            summary["obtained_marks"] = 0
            summary["percentage"] = 0
        return [], summary

    weights_by_step_id = _resolve_step_weights(step_results)
    total_weight = sum(weights_by_step_id.values()) or float(len(step_results))
    blocks_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for step_result in step_results:
        block_id = str(step_result.get("blockId", "")).strip() or "block_1"
        if block_id not in blocks_by_id:
            ordered_ids.append(block_id)
            blocks_by_id[block_id] = {
                "blockId": block_id,
                "blockLabel": str(step_result.get("blockLabel", "")).strip() or block_id,
                "question_part_label": str(step_result.get("question_part_label", "")).strip(),
                "question_part_text": str(step_result.get("question_part_text", "")).strip(),
                "steps": [],
            }
        blocks_by_id[block_id]["steps"].append(step_result)

    block_results = []
    for block_id in ordered_ids:
        block = blocks_by_id[block_id]
        block_steps = block["steps"]
        block_weight = sum(weights_by_step_id[step["stepId"]] for step in block_steps)
        block_earned = sum(
            weights_by_step_id[step["stepId"]] * _status_credit(step["step_status"])
            for step in block_steps
        )
        block_payload = {
            "blockId": block["blockId"],
            "blockLabel": block["blockLabel"],
            "question_part_label": block["question_part_label"],
            "question_part_text": block["question_part_text"],
            "status": _aggregate_status([step["step_status"] for step in block_steps]),
            "status_breakdown": _summarize_status_counts(block_steps),
            "weight": round(block_weight, 3),
            "steps": block_steps,
        }
        if full_marks is not None:
            block_payload["max_marks"] = _clean_marks_value(full_marks * (block_weight / total_weight))
            block_payload["obtained_marks"] = _clean_marks_value(full_marks * (block_earned / total_weight))
        block_results.append(block_payload)

    overall_earned = sum(
        weights_by_step_id[step["stepId"]] * _status_credit(step["step_status"])
        for step in step_results
    )
    obtained_marks = (full_marks * overall_earned / total_weight) if full_marks is not None else None

    summary = {
        "overall_status": _aggregate_status([step["step_status"] for step in step_results]),
        "status_breakdown": _summarize_status_counts(step_results),
        "step_count": len(step_results),
        "block_count": len(block_results),
        "total_weight": round(total_weight, 3),
    }
    if full_marks is not None and obtained_marks is not None:
        summary["full_marks"] = _clean_marks_value(full_marks)
        summary["obtained_marks"] = _clean_marks_value(obtained_marks)
        summary["percentage"] = _round_score((obtained_marks / full_marks) * 100)

    return block_results, summary


def _pick_block_topic(block_steps: list[dict[str, Any]]) -> str:
    topics: dict[str, int] = {}
    for step in block_steps:
        topic = _normalize_whitespace(str(step.get("topic", "")))
        if not topic:
            continue
        topics[topic] = topics.get(topic, 0) + 1

    if not topics:
        return "General"

    return max(topics, key=lambda topic: (topics[topic], len(topic)))


def _pick_block_understanding(block_steps: list[dict[str, Any]]) -> str:
    understandings = []
    seen = set()
    for step in block_steps:
        understanding = _normalize_whitespace(str(step.get("step_understanding", "")))
        if not understanding or understanding in seen:
            continue
        seen.add(understanding)
        understandings.append(understanding)

    if not understandings:
        return "Step intent is not fully clear."

    return " ".join(understandings)


def _pick_block_description(
    block_steps: list[dict[str, Any]],
    block_status: str,
) -> str:
    if block_status == "right":
        return ""

    preferred_status_order = [block_status]
    if block_status == "wrong":
        preferred_status_order.extend(["incomplete", "unknown"])
    elif block_status == "incomplete":
        preferred_status_order.append("unknown")

    for status in preferred_status_order:
        for step in block_steps:
            if step.get("step_status") != status:
                continue
            description = _normalize_whitespace(str(step.get("description", "")))
            if description:
                return description

    return ""


def _build_block_text(block_steps: list[dict[str, Any]]) -> str:
    texts = []
    seen = set()
    for step in block_steps:
        text = _normalize_whitespace(str(step.get("text", "")))
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)

    return "\n".join(texts)


def _build_public_response(
    block_results: list[dict[str, Any]],
    total_weight: float,
) -> list[dict[str, Any]]:
    public_response = []
    safe_total_weight = total_weight if total_weight > 0 else float(len(block_results) or 1)

    for index, block in enumerate(block_results, start=1):
        block_steps = list(block.get("steps", []))
        block_weight = float(block.get("weight", 0.0))
        public_response.append(
            {
                "stepId": str(index),
                "text": _build_block_text(block_steps),
                "step_status": _coerce_step_status(block.get("status", "unknown")),
                "step_weight": round(block_weight / safe_total_weight, 3),
                "topic": _pick_block_topic(block_steps),
                "step_understanding": _pick_block_understanding(block_steps),
                "description": _pick_block_description(
                    block_steps,
                    _coerce_step_status(block.get("status", "unknown")),
                ),
            }
        )

    return public_response


def _finalize_step_result(
    *,
    question: str,
    question_profile: dict[str, Any],
    step: dict[str, Any],
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    step_text = _normalize_latex_text(str(raw_result.get("text", step["text"])).strip()) or step["text"]
    step_understanding = str(raw_result.get("step_understanding", "")).strip()
    topic = str(raw_result.get("topic", "")).strip()
    step_type = _resolve_step_type(
        raw_step_type=str(raw_result.get("step_type", "")).strip(),
        question_profile=question_profile,
        question=question,
        step_text=step_text,
        step_understanding=step_understanding,
        topic=topic,
    )
    step_status = _coerce_step_status(raw_result.get("step_status", "unknown"))

    return {
        "stepId": step["stepId"],
        "sourceStepIds": step.get("sourceStepIds", [step["stepId"]]),
        "blockId": str(step.get("blockId", "")).strip(),
        "blockLabel": str(step.get("blockLabel", "")).strip(),
        "question_part_label": str(step.get("question_part_label", "")).strip(),
        "question_part_text": str(step.get("question_part_text", "")).strip(),
        "text": step_text,
        "step_status": step_status,
        "step_weight": _clamp_step_weight(raw_result.get("step_weight", 0.5)),
        "step_type": step_type,
        "topic": _coerce_topic(topic, step_type),
        "step_understanding": _coerce_step_understanding(step_understanding),
        "description": _coerce_description(
            step_status,
            str(raw_result.get("description", "")).strip(),
            step_type,
        ),
    }


def _evaluate_reconstructed_steps(
    *,
    question: str,
    steps: list[dict[str, Any]],
    question_profile: dict[str, Any],
    context: str = "",
    question_parts: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if not steps:
        return []

    solution_profile = _build_solution_profile(
        question=question,
        steps=steps,
        question_profile=question_profile,
        context=context,
        question_parts=question_parts,
    )
    parsed_response = _generate_json_response(
        model=GEMINI_CHAT_MODEL,
        system_instruction=STEP_EVALUATION_SYSTEM_INSTRUCTION,
        user_content=build_step_evaluation_prompt(
            question=question,
            question_parts=question_parts or [],
            question_profile=question_profile,
            solution_profile=solution_profile,
            solution_blocks=_serialize_blocks_for_prompt(steps),
            reconstructed_steps=steps,
            retrieved_context=context,
            allowed_step_types=CATEGORY_KEYS,
        ),
        schema={
            "type": "object",
            "properties": {
                "response": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stepId": {"type": "string"},
                            "text": {"type": "string"},
                            "step_status": {
                                "type": "string",
                                "enum": sorted(STATUS_VALUES),
                            },
                            "step_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "step_type": {
                                "type": "string",
                                "enum": CATEGORY_KEYS,
                            },
                            "topic": {"type": "string"},
                            "step_understanding": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "stepId",
                            "text",
                            "step_status",
                            "step_weight",
                            "step_type",
                            "topic",
                            "step_understanding",
                            "description",
                        ],
                    },
                }
            },
            "required": ["response"],
        },
    )

    raw_results = parsed_response.get("response", []) if isinstance(parsed_response, dict) else []
    results_by_step_id = {
        str(item.get("stepId", "")).strip(): item
        for item in raw_results
        if isinstance(item, dict)
    }

    final_results = []
    for index, step in enumerate(steps):
        finalized_step = _finalize_step_result(
            question=question,
            question_profile=question_profile,
            step=step,
            raw_result=results_by_step_id.get(step["stepId"], {}),
        )
        final_results.append(
            _post_process_step_result(
                question=question,
                steps=steps,
                index=index,
                step_result=finalized_step,
            )
        )
    return final_results


def _build_response_payload(
    *,
    steps: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
    full_marks: float | None,
    response_source: str | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    block_results, summary = _build_block_results(final_results, full_marks)
    payload = {
        "response": _build_public_response(
            block_results=block_results,
            total_weight=float(summary.get("total_weight", 0.0)),
        ),
    }
    if response_source is not None:
        payload["response_source"] = response_source
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    return payload


def evaluate_ocr_steps(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    full_marks: float | None = None,
):
    normalized_steps = _normalize_ocr_steps(ocr_data)
    logger.info("Evaluating %s OCR steps", len(normalized_steps))
    normalized_question = _normalize_latex_text(question.strip())
    reconstructed_steps = _reconstruct_solution_steps(normalized_steps, normalized_question)
    logger.info("Reconstructed OCR into %s logical steps", len(reconstructed_steps))
    question_parts = _extract_question_parts(normalized_question)
    annotated_steps, _ = _build_blocks_from_steps(reconstructed_steps, question_parts)
    question_profile = _build_question_profile(normalized_question)
    final_results = _evaluate_reconstructed_steps(
        question=normalized_question,
        steps=annotated_steps,
        question_profile=question_profile,
        question_parts=question_parts,
    )
    logger.info("OCR evaluation completed")
    return _build_response_payload(
        steps=annotated_steps,
        final_results=final_results,
        full_marks=_coerce_full_marks(full_marks),
        response_source="llm",
    )


def evaluate_ocr_steps_with_rag(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
    full_marks: float | None = None,
):
    normalized_steps = _normalize_ocr_steps(ocr_data)
    logger.info("Evaluating %s OCR steps with RAG", len(normalized_steps))
    normalized_question = _normalize_latex_text(question.strip())
    reconstructed_steps = _reconstruct_solution_steps(normalized_steps, normalized_question)
    logger.info("Reconstructed OCR into %s logical steps", len(reconstructed_steps))
    question_parts = _extract_question_parts(normalized_question)
    annotated_steps, _ = _build_blocks_from_steps(reconstructed_steps, question_parts)
    question_profile = _build_question_profile(normalized_question)
    context, fallback_reason = _get_rag_context(
        question=normalized_question,
        steps=annotated_steps,
        collection_name=collection_name,
        top_k=top_k,
    )
    final_results = _evaluate_reconstructed_steps(
        question=normalized_question,
        steps=annotated_steps,
        question_profile=question_profile,
        context=context,
        question_parts=question_parts,
    )
    logger.info("RAG-backed OCR evaluation completed")
    return _build_response_payload(
        steps=annotated_steps,
        final_results=final_results,
        full_marks=_coerce_full_marks(full_marks),
        response_source="rag" if context else "llm",
        fallback_reason=fallback_reason,
    )
