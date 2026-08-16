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
    RAG_MAX_CONTEXT_CHARS,
    RAG_MIN_SCORE,
    _apply_deterministic_step_checks,
    _decimal_places,
    _extract_assigned_variable,
    _extract_assigned_numeric_value,
    _extract_last_number,
    _extract_last_number_text,
    _format_number,
    _formula_checker,
    _generate_json_response,
    _normalize_latex_text,
    _validate_collection_name,
    index_documents,
    index_text_documents,
    retrieve_relevant_chunks,
)


STATUS_VALUES = {"right", "wrong", "unknown", "incomplete"}
RESPONSE_SCHEMA_VERSION = "1.0"
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
    used_step_ids: set[str] = set()
    for index, step in enumerate(ocr_data, start=1):
        if not isinstance(step, dict):
            logger.warning("Ignoring non-object OCR step at index %s", index)
            continue
        step_id = str(step.get("stepId", index)).strip() or str(index)
        if step_id in used_step_ids:
            base_step_id = step_id
            suffix = 2
            while f"{base_step_id}_{suffix}" in used_step_ids:
                suffix += 1
            step_id = f"{base_step_id}_{suffix}"
        used_step_ids.add(step_id)
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

    try:
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
    except Exception as exc:
        logger.warning("OCR reconstruction failed; preserving original steps: %s", exc)
        parsed_response = {}

    reconstructed_steps = parsed_response.get("steps", []) if isinstance(parsed_response, dict) else []
    source_by_id = {step["stepId"]: step for step in filtered_steps}
    source_position = {
        step["stepId"]: index
        for index, step in enumerate(filtered_steps)
    }
    used_source_ids: set[str] = set()
    grouped_steps: list[tuple[int, dict[str, Any]]] = []

    for step in reconstructed_steps:
        if not isinstance(step, dict) or not isinstance(step.get("sourceStepIds"), list):
            continue
        source_step_ids = []
        for raw_step_id in step["sourceStepIds"]:
            source_step_id = str(raw_step_id).strip()
            if (
                source_step_id
                and source_step_id in source_by_id
                and source_step_id not in used_source_ids
                and source_step_id not in source_step_ids
            ):
                source_step_ids.append(source_step_id)
        source_step_ids.sort(key=source_position.__getitem__)
        positions = [source_position[step_id] for step_id in source_step_ids]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            continue

        # The model may decide grouping, but it must never rewrite student work.
        # Rebuild text exclusively from the original OCR lines.
        grouped_steps.append(
            (
                positions[0],
                {
                    "stepId": source_step_ids[0],
                    "sourceStepIds": source_step_ids,
                    "text": " ".join(source_by_id[step_id]["text"] for step_id in source_step_ids),
                },
            )
        )
        used_source_ids.update(source_step_ids)

    for source_step in filtered_steps:
        if source_step["stepId"] in used_source_ids:
            continue
        grouped_steps.append(
            (
                source_position[source_step["stepId"]],
                {
                    "stepId": source_step["stepId"],
                    "sourceStepIds": [source_step["stepId"]],
                    "text": source_step["text"],
                },
            )
        )

    if grouped_steps:
        return [step for _, step in sorted(grouped_steps, key=lambda item: item[0])]

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

    has_part_labels = any(_extract_part_label_from_text(step["text"]) for step in steps)
    group_by_part = bool(question_parts or has_part_labels)

    for index, step in enumerate(steps, start=1):
        detected_label = _extract_part_label_from_text(step["text"])
        if current_block is None or detected_label is not None or not group_by_part:
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
) -> tuple[str, str | None, list[dict[str, Any]]]:
    try:
        relevant_chunks = retrieve_relevant_chunks(
            query=_build_retrieval_query(question, steps),
            collection_name=collection_name,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return "", "retrieval_error", []

    if not relevant_chunks:
        return "", "no_context", []

    grounded_chunks = [
        chunk
        for chunk in relevant_chunks
        if isinstance(chunk.get("score"), (int, float)) and chunk["score"] >= RAG_MIN_SCORE
    ]
    if not grounded_chunks:
        return "", "context_issue", []

    context_parts = []
    sources = []
    remaining_chars = max(0, RAG_MAX_CONTEXT_CHARS)
    for rank, chunk in enumerate(grounded_chunks, start=1):
        chunk_text = _normalize_whitespace(str(chunk.get("text", "")))
        if not chunk_text or remaining_chars <= 0:
            continue
        chunk_text = chunk_text[:remaining_chars]
        remaining_chars -= len(chunk_text)
        context_parts.append(
            f"[Source {rank} | document={chunk.get('document_id')} | "
            f"chunk={chunk.get('chunk_index')} | score={float(chunk['score']):.4f}]\n{chunk_text}"
        )
        sources.append(
            {
                "rank": rank,
                "score": round(float(chunk["score"]), 4),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "metadata": chunk.get("metadata", {}),
            }
        )

    context = "\n\n".join(context_parts)
    return context, None if context else "context_issue", sources


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


def _unit_for_quantity_text(text: str) -> set[str]:
    lowered = (text or "").lower()
    candidates = set()
    rules = [
        ("N", r"\b(?:force|friction|tension|thrust|weight)\b"),
        ("m/s^2", r"\bacceleration\b"),
        ("m/s", r"\b(?:velocity|speed)\b"),
        ("kg", r"\bmass\b"),
        ("m", r"\b(?:distance|displacement|length|height)\b"),
        ("s", r"\btime\b"),
    ]
    for unit, pattern in rules:
        if re.search(pattern, lowered):
            candidates.add(unit)
    return candidates


def _infer_expected_unit_from_context(
    question: str,
    local_context: str,
    step_text: str,
) -> str | None:
    variable = (_extract_assigned_variable(step_text) or "").lower()
    variable_units = {
        "f": "N",
        "a": "m/s^2",
        "u": "m/s",
        "v": "m/s",
        "m": "kg",
        "s": "m",
        "d": "m",
        "h": "m",
        "t": "s",
    }
    step_units = _unit_for_quantity_text(_strip_latex_commands(step_text))
    if len(step_units) == 1:
        return step_units.pop()
    nearby_units = _unit_for_quantity_text(_strip_latex_commands(local_context))
    if len(nearby_units) == 1:
        return nearby_units.pop()

    question_units = _unit_for_quantity_text(question)
    contextual_units = nearby_units | question_units
    if variable in variable_units and variable_units[variable] in contextual_units:
        return variable_units[variable]
    return question_units.pop() if len(question_units) == 1 else None


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
    reported_value = (
        _extract_assigned_numeric_value(step_text, variable)
        if variable
        else _extract_last_number(step_text)
    )
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
        step_result["step_weight"] = 0.0
        step_result["counts_toward_score"] = False
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
    step_result["counts_toward_score"] = step_result["step_weight"] > 0
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


def _counts_toward_score(step: dict[str, Any]) -> bool:
    if "counts_toward_score" in step:
        return bool(step["counts_toward_score"])
    return _clamp_step_weight(step.get("step_weight", 0.0)) > 0


def _build_public_steps(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stepId": str(step["stepId"]),
            "text": str(step.get("text", "")),
            "step_status": _coerce_step_status(step.get("step_status", "unknown")),
            "counts_toward_score": _counts_toward_score(step),
            "step_weight": _clamp_step_weight(step.get("step_weight", 0.0)),
            "step_type": str(step.get("step_type", STEP_TYPE_DEFAULT)),
            "topic": str(step.get("topic", "General")),
            "step_understanding": str(
                step.get("step_understanding", "Step intent is not fully clear.")
            ),
            "description": str(step.get("description", "")),
        }
        for step in step_results
    ]


def _build_summary(
    step_results: list[dict[str, Any]],
    full_marks: float | None,
) -> dict[str, Any]:
    scored_steps = [
        step
        for step in step_results
        if _counts_toward_score(step)
        and float(step.get("step_weight", 0.0)) > 0
    ]
    total_weight = sum(float(step["step_weight"]) for step in scored_steps)
    earned_weight = sum(
        float(step["step_weight"]) * _status_credit(step["step_status"])
        for step in scored_steps
    )
    percentage = (earned_weight / total_weight) * 100 if total_weight > 0 else 0.0
    summary: dict[str, Any] = {
        "overall_status": (
            _aggregate_status([step["step_status"] for step in scored_steps])
            if scored_steps
            else "unknown"
        ),
        "step_count": len(step_results),
        "scored_step_count": len(scored_steps),
        "percentage": _round_score(percentage),
        "status_breakdown": _summarize_status_counts(step_results),
    }
    if full_marks is not None:
        summary["full_marks"] = _clean_marks_value(full_marks)
        summary["obtained_marks"] = _clean_marks_value(
            full_marks * earned_weight / total_weight if total_weight > 0 else 0
        )
    return summary


def _finalize_step_result(
    *,
    question: str,
    question_profile: dict[str, Any],
    step: dict[str, Any],
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    # Model output is judgment only. The student text is immutable evidence and
    # must not be replaced by a model-corrected version.
    step_text = _normalize_latex_text(str(step["text"]).strip())
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
    final_results: list[dict[str, Any]],
    full_marks: float | None,
    grounding_status: str,
    grounding_reason: str | None = None,
    rag_sources: list[dict[str, Any]] | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "steps": _build_public_steps(final_results),
        "summary": _build_summary(final_results, full_marks),
        "grounding": {
            "status": grounding_status,
            "collection_name": collection_name,
            "reason": grounding_reason,
            "sources": rag_sources or [],
        },
    }


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
        final_results=final_results,
        full_marks=_coerce_full_marks(full_marks),
        grounding_status="not_requested",
    )


def evaluate_ocr_steps_with_rag(
    ocr_data: list[dict[str, Any]],
    question: str = "",
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
    full_marks: float | None = None,
):
    collection_name = _validate_collection_name(collection_name)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    normalized_steps = _normalize_ocr_steps(ocr_data)
    logger.info("Evaluating %s OCR steps with RAG", len(normalized_steps))
    normalized_question = _normalize_latex_text(question.strip())
    reconstructed_steps = _reconstruct_solution_steps(normalized_steps, normalized_question)
    logger.info("Reconstructed OCR into %s logical steps", len(reconstructed_steps))
    question_parts = _extract_question_parts(normalized_question)
    annotated_steps, _ = _build_blocks_from_steps(reconstructed_steps, question_parts)
    question_profile = _build_question_profile(normalized_question)
    context, fallback_reason, rag_sources = _get_rag_context(
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
        final_results=final_results,
        full_marks=_coerce_full_marks(full_marks),
        grounding_status="used" if context else "fallback",
        grounding_reason=fallback_reason,
        rag_sources=rag_sources,
        collection_name=collection_name,
    )
