# Evaluation Engine API

The repository application mounts this engine at:

```text
/evaluation_engine
```

Run the combined service from `engines/` with:

```bash
PYTHONPATH=. .venv/bin/python app.py
```

## Health check

```http
GET /evaluation_engine/
```

## OCR transcription

```http
POST /evaluation_engine/get_json_ocr
Content-Type: application/json
```

```json
{
  "source": "/absolute/path/to/solution.jpg"
}
```

Returns OCR-preserved steps under `ocr_data`.

Local image paths must resolve inside `OCR_LOCAL_FILE_ROOT` (the repository root by
default), use JPEG/PNG/WEBP/GIF, and stay within `OCR_MAX_IMAGE_BYTES`.

## Evaluate OCR steps

```http
POST /evaluation_engine/checked_json_ocr
Content-Type: application/json
```

Provide either `ocr_data` or an image `source`/`solution_url`:

```json
{
  "source": "/absolute/path/to/solution.jpg",
  "question": "7x - 2 = 2(11x + 5). Find x.",
  "full_marks": 5
}
```

## Evaluate OCR steps with RAG

```http
POST /evaluation_engine/checked_json_ocr_with_rag
Content-Type: application/json
```

```json
{
  "source": "/absolute/path/to/solution.jpg",
  "question": "7x - 2 = 2(11x + 5). Find x.",
  "collection_name": "openai_docs",
  "top_k": 3,
  "full_marks": 5
}
```

The response uses the flat evaluation contract documented below.

## Evaluation status

```http
POST /evaluation_engine/evaluated_json_ocr
```

Returns a simple service acknowledgement.

## Analyze a solution image

```http
POST /evaluation_engine/get_analysis
Content-Type: application/json
```

```json
{
  "image_source": "/absolute/path/to/solution.jpg",
  "question": "Enter the original question here.",
  "collection_name": "evaluation_engine",
  "top_k": 5,
  "full_marks": 5
}
```

This main application flow is RAG-backed. If Qdrant is unavailable or no retrieved
chunk meets `RAG_MIN_SCORE`, evaluation continues with the question and student work,
and returns `grounding.status: "fallback"` plus a stable `grounding.reason`.

## Evaluation response

Evaluation endpoints return one flat, versioned contract:

- `steps`: logical step judgments with original OCR evidence preserved.
- `summary`: counts and a weighted percentage; marks are included when `full_marks` is supplied.
- `grounding`: `used`, `fallback`, or `not_requested`, plus sources supplied to the evaluator.

`counts_toward_score` is false for copied questions, headings, labels, and irrelevant work.
`sourceStepIds` and multipart block metadata remain internal and are not exposed publicly.

```json
{
  "schema_version": "1.0",
  "steps": [
    {
      "stepId": "1",
      "text": "5x - 2 = 22x + 10",
      "step_status": "right",
      "counts_toward_score": true,
      "step_weight": 0.3,
      "step_type": "calculation_based",
      "topic": "Linear Equations",
      "step_understanding": "The student correctly expands the equation.",
      "description": ""
    }
  ],
  "summary": {
    "overall_status": "right",
    "step_count": 1,
    "scored_step_count": 1,
    "percentage": 100,
    "status_breakdown": {
      "right": 1,
      "wrong": 0,
      "incomplete": 0,
      "unknown": 0
    }
  },
  "grounding": {
    "status": "used",
    "collection_name": "evaluation_engine",
    "reason": null,
    "sources": [
      {
        "rank": 1,
        "score": 0.92,
        "document_id": "algebra-chapter-2",
        "chunk_index": 3,
        "metadata": {"chapter": "Linear Equations"}
      }
    ]
  }
}
```

`top_k` must be an integer from 1 to 20. `full_marks`, when provided, must be positive.

## Index RAG documents

```http
POST /evaluation_engine/index_documents
Content-Type: application/json
```

```json
{
  "documents": [
    {"text": "NCERT reference content", "metadata": {"source": "ncert"}}
  ],
  "collection_name": "openai_docs"
}
```

For local text files, use `POST /evaluation_engine/index_text_documents` with:

```json
{
  "document_paths": ["/absolute/path/to/reference.txt"],
  "collection_name": "openai_docs"
}
```

Local RAG files must be `.txt` or `.md`, resolve inside `RAG_DOCUMENT_ROOT` (`Data/`
by default), and stay within `RAG_MAX_DOCUMENT_BYTES`.

All validation failures return HTTP `400`; processing failures return HTTP `500` with a
generic `error` field. Detailed provider errors are logged server-side and are not exposed
to clients.
