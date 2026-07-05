import re


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def normalize_chapter_name(chapter: str) -> str:
    normalized = normalize_text(chapter)
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", normalized).strip("_")
