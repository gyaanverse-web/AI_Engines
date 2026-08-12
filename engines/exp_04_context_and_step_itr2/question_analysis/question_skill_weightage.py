from .preprocessor import normalize_text
from .scorer import (
    calculate_question_skill_weightage,
    calculate_question_skill_weightage_with_meta as calculate_question_skill_weightage_with_meta_result,
)
from .validator import validate_question_analysis_request


def analyze_question_skill_weightage(payload: object, use_ml: bool = True) -> dict[str, int]:
    validated_payload = validate_question_analysis_request(payload)
    normalized_question = normalize_text(validated_payload["question"])
    normalized_chapter = normalize_text(validated_payload["chapter"])
    resolved_use_ml = validated_payload["use_ml"]
    if resolved_use_ml is None:
        resolved_use_ml = use_ml

    return calculate_question_skill_weightage(
        question=normalized_question,
        chapter=normalized_chapter,
        use_ml=resolved_use_ml,
    )


def analyze_question_skill_weightage_with_meta(
    payload: object,
    use_ml: bool = True,
) -> dict[str, object]:
    validated_payload = validate_question_analysis_request(payload)
    normalized_question = normalize_text(validated_payload["question"])
    normalized_chapter = normalize_text(validated_payload["chapter"])
    resolved_use_ml = validated_payload["use_ml"]
    if resolved_use_ml is None:
        resolved_use_ml = use_ml

    return calculate_question_skill_weightage_with_meta_result(
        question=normalized_question,
        chapter=normalized_chapter,
        use_ml=resolved_use_ml,
    )
