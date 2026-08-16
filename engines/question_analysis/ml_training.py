import json
from collections import Counter, defaultdict
from pathlib import Path

from .categories import CATEGORY_KEYS
from .config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from .feature_extractor import extract_features
from .preprocessor import normalize_text
from .tokenizer import tokenize_text


def _load_dataset_records(dataset_path: Path) -> list[dict[str, object]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def train_model_from_dataset(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
) -> Path:
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    model_path = model_path or DEFAULT_MODEL_PATH

    records = _load_dataset_records(dataset_path)
    if not records:
        raise ValueError("Training dataset is empty")

    chapter_totals = defaultdict(lambda: {category: 0.0 for category in CATEGORY_KEYS})
    chapter_counts = Counter()
    token_totals = defaultdict(lambda: {category: 0.0 for category in CATEGORY_KEYS})
    token_counts = Counter()
    feature_totals = defaultdict(lambda: {category: 0.0 for category in CATEGORY_KEYS})
    feature_counts = Counter()
    global_totals = {category: 0.0 for category in CATEGORY_KEYS}

    for record in records:
        question = normalize_text(str(record["question"]))
        chapter = normalize_text(str(record["chapter"]))
        labels = {category: float(record["labels"].get(category, 0.0)) for category in CATEGORY_KEYS}

        for category in CATEGORY_KEYS:
            chapter_totals[chapter][category] += labels[category]
            global_totals[category] += labels[category]
        chapter_counts[chapter] += 1

        for token in set(tokenize_text(f"{question} {chapter}")):
            for category in CATEGORY_KEYS:
                token_totals[token][category] += labels[category]
            token_counts[token] += 1

        features = extract_features(question=question, chapter=chapter)
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, bool):
                numeric_value = 1 if feature_value else 0
            elif isinstance(feature_value, (int, float)):
                numeric_value = float(feature_value)
            else:
                numeric_value = len(feature_value) if isinstance(feature_value, list) else 0

            if numeric_value <= 0:
                continue

            feature_key = f"{feature_name}:{numeric_value}"
            for category in CATEGORY_KEYS:
                feature_totals[feature_key][category] += labels[category]
            feature_counts[feature_key] += 1

    record_count = len(records)
    global_average = {
        category: global_totals[category] / record_count
        for category in CATEGORY_KEYS
    }

    chapter_profiles = {
        chapter: {
            category: chapter_totals[chapter][category] / chapter_counts[chapter]
            for category in CATEGORY_KEYS
        }
        for chapter in chapter_counts
    }
    token_profiles = {
        token: {
            category: token_totals[token][category] / token_counts[token]
            for category in CATEGORY_KEYS
        }
        for token in token_counts
        if token_counts[token] >= 2
    }
    feature_profiles = {
        feature_key: {
            category: feature_totals[feature_key][category] / feature_counts[feature_key]
            for category in CATEGORY_KEYS
        }
        for feature_key in feature_counts
    }

    model_artifact = {
        "model_type": "bootstrap_keyword_profile_v1",
        "record_count": record_count,
        "global_average": global_average,
        "chapter_profiles": chapter_profiles,
        "token_profiles": token_profiles,
        "feature_profiles": feature_profiles,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as model_file:
        json.dump(model_artifact, model_file, ensure_ascii=False, indent=2)

    return model_path
