import logging
import time
from typing import Any


logger = logging.getLogger(__name__)


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
    started_at = time.perf_counter()
    from .ocr_engine import extract_ocr_steps
    from .provider_engine import _validate_collection_name
    from .testing_engine import evaluate_ocr_steps, evaluate_ocr_steps_with_rag

    if use_rag:
        collection_name = _validate_collection_name(collection_name)

    logger.info(
        "Solution analysis started mode=%s ocr_input=%s question_provided=%s",
        "rag" if use_rag else "standard",
        "provided" if ocr_data is not None else "image",
        bool(question.strip()),
    )
    if ocr_data is None:
        if not image_source:
            raise ValueError("image_source or ocr_data is required")
        ocr_started_at = time.perf_counter()
        logger.info("OCR extraction started for solution analysis")
        ocr_data = extract_ocr_steps(image_source)
        logger.info(
            "OCR extraction completed steps=%s duration_ms=%.1f",
            len(ocr_data),
            (time.perf_counter() - ocr_started_at) * 1000,
        )
    else:
        logger.info("Using provided OCR data steps=%s", len(ocr_data))

    logger.info("Dispatching %s OCR steps for evaluation", len(ocr_data))
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
    logger.info(
        "Solution analysis completed result_steps=%s duration_ms=%.1f",
        len(evaluated_response.get("steps", [])),
        (time.perf_counter() - started_at) * 1000,
    )
    return evaluated_response


# Backward-compatible alias for existing integrations.
analyzer = analyze_solution
