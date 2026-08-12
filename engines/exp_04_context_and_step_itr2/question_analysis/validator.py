def validate_question_analysis_request(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    question = payload.get("question")
    chapter = payload.get("chapter")

    if question is None:
        raise ValueError("question is required")
    if chapter is None:
        raise ValueError("chapter is required")
    if not isinstance(question, str):
        raise ValueError("question must be a non-empty string")
    if not isinstance(chapter, str):
        raise ValueError("chapter must be a non-empty string")
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if not chapter.strip():
        raise ValueError("chapter must be a non-empty string")

    use_ml = payload.get("use_ml")
    if use_ml is not None and not isinstance(use_ml, bool):
        raise ValueError("use_ml must be a boolean")

    return {
        "question": question,
        "chapter": chapter,
        "use_ml": use_ml,
    }
