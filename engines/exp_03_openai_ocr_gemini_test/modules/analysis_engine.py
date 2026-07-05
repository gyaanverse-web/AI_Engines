from typing import Any

from analysis_engine import analyze_question_skill_weightage


def analyzer(
    image_source: str = "",
    question: str = "",
    collection_name: str | None = None,
    top_k: int = 5,
    ocr_data: list[dict[str, Any]] | None = None,
):
    from .ocr_engine import get_json_ocr
    from .testing_engine import evaluate_ocr_steps as run_evaluate_ocr_steps

    if ocr_data is None:
        if not image_source:
            raise ValueError("image_source or ocr_data is required")
        ocr_data = get_json_ocr(image_source)

    print(f"[analysis_engine.analyzer] OCR returned {len(ocr_data)} steps")

    evaluated_response = run_evaluate_ocr_steps(
        ocr_data=ocr_data,
        question=question,
    )
    print("[analysis_engine.analyzer] Evaluation completed")

    return {
        "response": evaluated_response.get("response", evaluated_response),
    }
