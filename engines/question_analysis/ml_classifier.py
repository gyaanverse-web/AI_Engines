import json
from pathlib import Path

from .categories import CATEGORY_KEYS
from .config import get_model_path, is_ml_enabled
from .feature_extractor import extract_features
from .preprocessor import normalize_text
from .tokenizer import tokenize_text


class MLClassifierService:
    def __init__(self, enabled: bool | None = None, model_path: str | Path | None = None):
        self.enabled = is_ml_enabled() if enabled is None else enabled
        self.model_path = Path(model_path) if model_path else get_model_path()
        self.model_artifact = self._load_model() if self.enabled else None

    def is_enabled(self) -> bool:
        return self.enabled and self.model_artifact is not None

    def predict(self, features: dict[str, object]) -> dict[str, float] | None:
        if not self.is_enabled():
            return None

        prediction = self._predict_impl(features)
        return self.validate_prediction(prediction)

    def _predict_impl(self, features: dict[str, object]) -> dict[str, float]:
        global_average = self.model_artifact["global_average"]
        chapter_profiles = self.model_artifact["chapter_profiles"]
        token_profiles = self.model_artifact["token_profiles"]
        feature_profiles = self.model_artifact["feature_profiles"]

        chapter = normalize_text(str(features.get("normalized_chapter", "")))
        question = normalize_text(str(features.get("normalized_question", "")))

        category_scores = {category: float(global_average.get(category, 0.0)) for category in CATEGORY_KEYS}
        contribution_count = 1

        chapter_profile = chapter_profiles.get(chapter)
        if chapter_profile:
            for category in CATEGORY_KEYS:
                category_scores[category] += float(chapter_profile.get(category, 0.0))
            contribution_count += 1

        for token in set(tokenize_text(f"{question} {chapter}")):
            token_profile = token_profiles.get(token)
            if not token_profile:
                continue
            for category in CATEGORY_KEYS:
                category_scores[category] += float(token_profile.get(category, 0.0))
            contribution_count += 1

        for feature_key in self._feature_keys(features):
            feature_profile = feature_profiles.get(feature_key)
            if not feature_profile:
                continue
            for category in CATEGORY_KEYS:
                category_scores[category] += float(feature_profile.get(category, 0.0))
            contribution_count += 1

        return {
            category: category_scores[category] / contribution_count
            for category in CATEGORY_KEYS
        }

    def _load_model(self) -> dict[str, object] | None:
        if not self.model_path.exists():
            return None
        if self.model_path.is_dir():
            return None

        with self.model_path.open("r", encoding="utf-8") as model_file:
            return json.load(model_file)

    @staticmethod
    def validate_prediction(prediction: dict[str, float]) -> dict[str, float]:
        validated_prediction = {}

        for category in CATEGORY_KEYS:
            value = prediction.get(category, 0.0)
            validated_prediction[category] = max(0.0, float(value))

        return validated_prediction

    @staticmethod
    def _feature_keys(features: dict[str, object]) -> list[str]:
        extracted_feature_keys = []

        for feature_name, feature_value in features.items():
            if feature_name.startswith("normalized_"):
                continue

            if isinstance(feature_value, bool):
                numeric_value = 1 if feature_value else 0
            elif isinstance(feature_value, (int, float)):
                numeric_value = float(feature_value)
            elif isinstance(feature_value, list):
                numeric_value = len(feature_value)
            else:
                numeric_value = 0

            if numeric_value > 0:
                extracted_feature_keys.append(f"{feature_name}:{numeric_value}")

        return extracted_feature_keys


def build_ml_features(question: str, chapter: str) -> dict[str, object]:
    normalized_question = normalize_text(question)
    normalized_chapter = normalize_text(chapter)
    features = extract_features(
        question=normalized_question,
        chapter=normalized_chapter,
    )
    features["normalized_question"] = normalized_question
    features["normalized_chapter"] = normalized_chapter
    return features
