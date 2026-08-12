from .categories import CATEGORY_KEYS
from .chapter_mappings import CHAPTER_WEIGHT_MAPPINGS
from .config import get_model_weight, get_rule_weight
from .feature_extractor import extract_features
from .ml_classifier import MLClassifierService, build_ml_features
from .normalizer import normalize_scores
from .preprocessor import normalize_chapter_name
from .rules import QUESTION_ANALYSIS_RULES


def _empty_scores() -> dict[str, float]:
    return {category: 0.0 for category in CATEGORY_KEYS}


def _apply_chapter_mapping(scores: dict[str, float], chapter: str) -> None:
    chapter_key = normalize_chapter_name(chapter)
    mapping = CHAPTER_WEIGHT_MAPPINGS.get(chapter_key)
    if not mapping:
        return

    for category, value in mapping.items():
        scores[category] += value


def _apply_keyword_rules(scores: dict[str, float], question: str) -> None:
    for rule in QUESTION_ANALYSIS_RULES:
        match_count = sum(1 for keyword in rule["keywords"] if keyword in question)
        if match_count:
            scores[rule["category"]] += rule["score_boost"] * match_count


def _apply_feature_boosts(scores: dict[str, float], features: dict[str, object]) -> None:
    if features["has_numbers"]:
        scores["calculation_based"] += 15
    if features["has_percentage"]:
        scores["formula_based"] += 15
        scores["calculation_based"] += 10
    if features["has_currency"]:
        scores["calculation_based"] += 15
        scores["application_based"] += 5
    if features["has_units"]:
        scores["calculation_based"] += 10
        scores["formula_based"] += 10
    if features["has_math_operator"]:
        scores["calculation_based"] += 15
    if features["has_equation"]:
        scores["formula_based"] += 15
        scores["calculation_based"] += 10
    if features["question_length"] >= 20:
        scores["language_or_explanation_based"] += 10
    if features["command_words"]:
        scores["reasoning_based"] += min(10, len(features["command_words"]) * 2)
    if features["chapter_keywords"]:
        scores["concept_based"] += min(15, len(features["chapter_keywords"]) * 3)
        scores["formula_based"] += min(15, len(features["chapter_keywords"]) * 2)


def calculate_rule_scores(question: str, chapter: str) -> dict[str, float]:
    scores = _empty_scores()

    _apply_chapter_mapping(scores, chapter)
    features = extract_features(question=question, chapter=chapter)
    _apply_keyword_rules(scores, question)
    _apply_feature_boosts(scores, features)
    return scores


def calculate_question_skill_weightage(
    question: str,
    chapter: str,
    use_ml: bool = True,
    ml_service: MLClassifierService | None = None,
) -> dict[str, int]:
    return calculate_question_skill_weightage_with_meta(
        question=question,
        chapter=chapter,
        use_ml=use_ml,
        ml_service=ml_service,
    )["weights"]


def calculate_question_skill_weightage_with_meta(
    question: str,
    chapter: str,
    use_ml: bool = True,
    ml_service: MLClassifierService | None = None,
) -> dict[str, object]:
    scores = calculate_rule_scores(question=question, chapter=chapter)
    mode = "rule_only"

    if use_ml:
        ml_features = build_ml_features(question=question, chapter=chapter)
        ml_service = ml_service or MLClassifierService(enabled=True)
        ml_scores = ml_service.predict(ml_features)
    else:
        ml_scores = None

    if ml_scores:
        rule_weight = get_rule_weight()
        model_weight = get_model_weight()
        for category in CATEGORY_KEYS:
            scores[category] = (scores[category] * rule_weight) + (
                float(ml_scores.get(category, 0)) * model_weight
            )
        mode = "rule_plus_ml"

    return {
        "weights": normalize_scores(scores),
        "mode": mode,
    }
