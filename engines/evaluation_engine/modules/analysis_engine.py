from typing import Any


def analyze_solution(
    image_source: str = "",
    question: str = "",
    collection_name: str | None = None,
    top_k: int = 5,
    ocr_data: list[dict[str, Any]] | None = None,
    full_marks: float | None = None,
    use_rag: bool = True,
) -> dict[str, Any]:
    """Run OCR when needed and evaluate the resulting student solution."""
    from .ocr_engine import extract_ocr_steps
    from .provider_engine import _validate_collection_name
    from .testing_engine import evaluate_ocr_steps, evaluate_ocr_steps_with_rag

    if use_rag:
        collection_name = _validate_collection_name(collection_name)

    if ocr_data is None:
        if not image_source:
            raise ValueError("image_source or ocr_data is required")
        ocr_data = extract_ocr_steps(image_source)

    if use_rag:
        evaluated_response = evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            full_marks=full_marks,
        )
    else:
        evaluated_response = evaluate_ocr_steps(
            ocr_data=ocr_data,
            question=question,
            full_marks=full_marks,
        )
    if not isinstance(evaluated_response, dict):
        raise TypeError("evaluation engine returned an invalid response payload")
    return evaluated_response


# Backward-compatible alias for existing integrations.
analyzer = analyze_solution
