import re


TOKEN_PATTERN = re.compile(r"[a-z0-9%₹\^/\-]+")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())
