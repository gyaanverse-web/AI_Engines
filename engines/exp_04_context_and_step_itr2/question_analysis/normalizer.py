from .categories import CATEGORY_KEYS, SAFE_DEFAULT_WEIGHTS


def normalize_scores(raw_scores: dict[str, float]) -> dict[str, int]:
    sanitized_scores = {
        category: max(0.0, float(raw_scores.get(category, 0)))
        for category in CATEGORY_KEYS
    }
    total = sum(sanitized_scores.values())

    if total == 0:
        return dict(SAFE_DEFAULT_WEIGHTS)

    normalized = {
        category: int(round((score / total) * 100))
        for category, score in sanitized_scores.items()
    }

    difference = 100 - sum(normalized.values())
    if difference:
        highest_category = max(
            CATEGORY_KEYS,
            key=lambda category: (sanitized_scores[category], -CATEGORY_KEYS.index(category)),
        )
        normalized[highest_category] += difference

    return {category: normalized.get(category, 0) for category in CATEGORY_KEYS}
