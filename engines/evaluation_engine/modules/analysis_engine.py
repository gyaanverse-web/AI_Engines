from typing import Any


def analyze_solution(
    image_source: str = "",
    question: str = "",
    collection_name: str | None = None,
    top_k: int = 5,
    ocr_data: list[dict[str, Any]] | None = None,
    full_marks: float | None = None,
) -> dict[str, Any]:
    """Run OCR when needed and evaluate the resulting student solution."""
    from .ocr_engine import extract_ocr_steps
    from .testing_engine import evaluate_ocr_steps

    if ocr_data is None:
        if not image_source:
            raise ValueError("image_source or ocr_data is required")
        ocr_data = extract_ocr_steps(image_source)

    evaluated_response = evaluate_ocr_steps(
        ocr_data=ocr_data,
        question=question,
        full_marks=full_marks,
    )
    if isinstance(evaluated_response, dict):
        return evaluated_response

    return {
        "response": evaluated_response,
    }


# Backward-compatible alias for existing integrations.
analyzer = analyze_solution
