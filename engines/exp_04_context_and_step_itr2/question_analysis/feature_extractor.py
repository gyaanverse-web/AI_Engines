import re


COMMAND_WORDS = {
    "calculate",
    "find",
    "solve",
    "determine",
    "evaluate",
    "compute",
    "simplify",
    "explain",
    "define",
    "prove",
    "derive",
    "compare",
    "justify",
    "draw",
    "describe",
    "state",
    "list",
}

CHAPTER_KEYWORDS = {
    "simple interest": ["interest", "principal", "rate", "amount", "per annum"],
    "compound interest": ["interest", "principal", "rate", "amount", "annually"],
    "mensuration": ["area", "volume", "perimeter", "circumference"],
    "photosynthesis": ["photosynthesis", "chlorophyll", "sunlight", "carbon dioxide"],
    "grammar": ["tense", "noun", "verb", "adjective", "grammar"],
    "proof": ["prove", "show that", "derive", "theorem"],
    "data interpretation": ["table", "graph", "chart", "data"],
}

UNIT_PATTERN = re.compile(
    r"\b(cm|mm|m|km|g|kg|mg|l|ml|s|sec|min|hr|hour|day|year|years|month|months|m/s|km/h)\b"
)
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
MATH_OPERATOR_PATTERN = re.compile(r"[\+\-\*/\^]")
EQUATION_PATTERN = re.compile(r"=")


def extract_features(question: str, chapter: str) -> dict[str, object]:
    tokens = question.split()
    number_matches = NUMBER_PATTERN.findall(question)
    matched_command_words = sorted({word for word in COMMAND_WORDS if word in question})
    matched_chapter_keywords = sorted(
        {
            keyword
            for chapter_name, keywords in CHAPTER_KEYWORDS.items()
            if chapter_name in chapter or chapter in chapter_name
            for keyword in keywords
            if keyword in question
        }
    )

    return {
        "number_count": len(number_matches),
        "has_numbers": bool(number_matches),
        "has_percentage": "%" in question or "percent" in question or "percentage" in question,
        "has_currency": "₹" in question or "rs" in question or "rupee" in question,
        "has_units": bool(UNIT_PATTERN.search(question)),
        "has_math_operator": bool(MATH_OPERATOR_PATTERN.search(question)),
        "has_equation": bool(EQUATION_PATTERN.search(question)),
        "question_length": len(tokens),
        "command_words": matched_command_words,
        "chapter_keywords": matched_chapter_keywords,
    }
