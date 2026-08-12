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

The response includes `response_source` (`rag` or `llm`) and a `response` array containing the evaluated steps.

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
  "full_marks": 5
}
```

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

All validation failures return HTTP `400`; processing failures return HTTP `500` with an `error` field.
