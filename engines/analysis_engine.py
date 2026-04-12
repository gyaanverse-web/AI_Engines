from ocr_engine import get_json_ocr
from testing_engine import evaluate_ocr_steps_with_rag


def analyzer(
    image_source: str,
    question: str = "",
    collection_name: str | None = None,
    top_k: int = 5,
):
    print(f"[analysis_engine.analyzer] Called with image_source: {image_source}")
    ocr_data = get_json_ocr(image_source)
    print(f"[analysis_engine.analyzer] OCR returned {len(ocr_data)} steps")

    if collection_name:
        evaluated_response = evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            collection_name=collection_name,
            top_k=top_k,
        )
    else:
        evaluated_response = evaluate_ocr_steps_with_rag(
            ocr_data=ocr_data,
            question=question,
            top_k=top_k,
        )
    print("[analysis_engine.analyzer] Evaluation completed")

    return {
        "evaluated_response": evaluated_response,
    }
