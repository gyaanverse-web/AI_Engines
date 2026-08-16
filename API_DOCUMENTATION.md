# GyaanVerse API Documentation

This document covers all API endpoints currently present in this repository.

## Services

| Service | Folder | Default Base URL | Status |
| --- | --- | --- | --- |
| Flask AI Engine | `engines` | `http://127.0.0.1:5000` | Active |
| SvelteKit API Proxy | `gyanverse` | Same origin as frontend | Active |
| Fastify SaaS Backend | `gyaanverse-software` | `http://127.0.0.1:3000` | Mostly scaffolded |

## Request Flow

Main implemented analysis flow:

```text
Browser
  -> POST /api/analyze-image
  -> SvelteKit proxy
  -> POST http://127.0.0.1:5000/evaluation_engine/get_analysis
  -> Flask analyzer
  -> OpenAI OCR
  -> Qdrant retrieval + grounded Gemini step evaluation
  -> JSON response
```

---

# 1. SvelteKit API Proxy

Base URL: same origin as the SvelteKit app.

## POST `/api/analyze-image`

Proxy endpoint used by the frontend. It forwards image analysis requests to the Flask engine
`/evaluation_engine/get_analysis` route.

### Request Body

```json
{
  "solution_url": "data:image/jpeg;base64,...",
  "question": "What is the work to be done...",
  "top_k": 5,
  "collection_name": "optional_collection"
}
```

### Accepted Image Fields

The endpoint accepts any one of these fields:

| Field | Description |
| --- | --- |
| `solution_url` | Preferred field. Can be a data URL, remote URL, or path understood by the Flask engine. |
| `image_source` | Alternate field name. |
| `answerImage` | Alternate field name. |

### Optional Fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `question` | string | `""` | Original question text. |
| `top_k` | number | `5` | Number of RAG chunks to retrieve; must be from 1 to 20. |
| `collection_name` | string | `ANALYSIS_COLLECTION_NAME` env var or engine default | Qdrant collection used for grounding. |

### Success Response

Returns the flat, versioned response documented under
`POST /evaluation_engine/get_analysis`.

### Error Responses

```json
{
  "error": "solution_url is required."
}
```

```json
{
  "error": "Failed to analyze image."
}
```

### Example

```bash
curl -X POST http://localhost:5173/api/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "solution_url": "data:image/jpeg;base64,...",
    "question": "Solve the given numerical.",
    "top_k": 5
  }'
```

---

# 2. Flask AI Engine APIs

Base URL: `http://127.0.0.1:5000`

Implemented in `engines/evaluation_engine/routes.py`.

## GET `/`

Health/welcome route.

### Success Response

```text
welcome to flask engine server
```

### Example

```bash
curl http://127.0.0.1:5000/
```

---

## POST `/evaluation_engine/get_json_ocr`

Runs OCR only and returns extracted solution steps.

### Request Body

```json
{
  "source": "data:image/jpeg;base64,..."
}
```

### `source` Values

| Type | Example |
| --- | --- |
| Data URL | `data:image/jpeg;base64,...` |
| Remote URL | `https://example.com/image.jpg` |
| Local file path | `Data/testImage4.jpeg` |

### Success Response

```json
{
  "ocr_data": [
    {
      "stepId": "1",
      "text": "Given m = 1500\\ \\mathrm{kg}"
    }
  ]
}
```

### Error Responses

```json
{
  "error": "source is required"
}
```

```json
{
  "error": "Image file not found: Data/missing.jpeg"
}
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/get_json_ocr \
  -H "Content-Type: application/json" \
  -d '{"source":"Data/testImage4.jpeg"}'
```

---

## POST `/evaluation_engine/get_analysis`

Runs full analysis from an image source. The backend performs OCR first, then evaluates each extracted step.

### Request Body

```json
{
  "image_source": "data:image/jpeg;base64,...",
  "question": "What is the work to be done...",
  "collection_name": "optional_collection",
  "top_k": 5
}
```

### Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `image_source` | string | Yes | Data URL, remote URL, or local image path. |
| `question` | string | No | Original question text. Improves evaluation context. |
| `collection_name` | string | No | Qdrant collection used for grounding. |
| `top_k` | number | No | RAG retrieval count from 1 to 20. |

### Success Response

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

`counts_toward_score` is false for copied questions, headings, labels, and irrelevant
work. Blocks, duplicate response arrays, and internal OCR `sourceStepIds` are not public.

### `step_status` Values

| Value | Meaning |
| --- | --- |
| `right` | Step is correct. |
| `wrong` | Step contains an error. |
| `incomplete` | Step is partially correct but missing something. |
| `unknown` | Evaluator could not confidently classify the step. |

### Error Responses

```json
{
  "error": "image_source is required"
}
```

```json
{
  "error": "..."
}
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/get_analysis \
  -H "Content-Type: application/json" \
  -d '{
    "image_source": "Data/testImage4.jpeg",
    "question": "What is the work to be done to increase the velocity of a car from 30 km/h to 60 km/h if the mass is 1500 kg?",
    "top_k": 5
  }'
```

---

## POST `/evaluation_engine/checked_json_ocr`

Runs evaluation using already extracted OCR data, or falls back to OCR if a solution image source is provided.

### Request Body With OCR Data

```json
{
  "ocr_data": [
    {
      "stepId": "1",
      "text": "Given m = 1500\\ \\mathrm{kg}"
    }
  ],
  "question": "What is the work to be done..."
}
```

### Request Body With Image Source

```json
{
  "solution_url": "Data/testImage4.jpeg",
  "question": "What is the work to be done..."
}
```

`source` may also be used instead of `solution_url`.

### Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ocr_data` | array | Conditional | Existing OCR steps. Required if no image source is provided. |
| `solution_url` | string | Conditional | Image source. Required if `ocr_data` is not provided. |
| `source` | string | Conditional | Alternate image source field. |
| `question` | string | No | Original question text. |

### Success Response

Same flat response format as `/evaluation_engine/get_analysis`, with
`grounding.status` set to `not_requested`.

### Error Response

```json
{
  "error": "Either ocr_data or solution_url is required"
}
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/checked_json_ocr \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_data": [
      {
        "stepId": "1",
        "text": "Given m = 1500\\ \\mathrm{kg}"
      }
    ],
    "question": "Find the work done."
  }'
```

---

## POST `/evaluation_engine/index_documents`

Indexes raw document text into Qdrant for retrieval.

### Request Body

```json
{
  "collection_name": "science_class_9",
  "documents": [
    {
      "document_id": "chapter_1",
      "text": "Document text to index...",
      "metadata": {
        "subject": "science",
        "class": "9"
      }
    }
  ]
}
```

### Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `documents` | array | Yes | Non-empty array of documents. |
| `collection_name` | string | No | Qdrant collection name. Defaults to `QDRANT_COLLECTION_NAME`. |
| `document_id` | string | No | Document identifier. Generated if omitted. |
| `text` | string | Yes | Text content to chunk and embed. |
| `metadata` | object | No | Stored with each indexed chunk. |

### Success Response

```json
{
  "collection_name": "science_class_9",
  "indexed_chunks": 12
}
```

### Error Response

```json
{
  "error": "documents must be a non-empty list"
}
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/index_documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "science_class_9",
    "documents": [
      {
        "document_id": "sample",
        "text": "Force is equal to mass times acceleration.",
        "metadata": {
          "source": "manual"
        }
      }
    ]
  }'
```

---

## POST `/evaluation_engine/index_text_documents`

Indexes local `.txt` files into Qdrant.

### Request Body

```json
{
  "collection_name": "maths_class_10",
  "document_paths": [
    "Data/s-chands-new-mathematics-class-x-school-books-9-12-2021_compress.txt"
  ]
}
```

### Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `document_paths` | array | Yes | Non-empty list of local text file paths. |
| `collection_name` | string | No | Qdrant collection name. Defaults to `QDRANT_COLLECTION_NAME`. |

### Success Response

```json
{
  "collection_name": "maths_class_10",
  "indexed_chunks": 120
}
```

### Error Responses

```json
{
  "error": "document_paths must be a non-empty list"
}
```

```json
{
  "error": "Text file not found: Data/missing.txt"
}
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/index_text_documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "maths_class_10",
    "document_paths": [
      "Data/s-chands-new-mathematics-class-x-school-books-9-12-2021_compress.txt"
    ]
  }'
```

---

## POST `/evaluation_engine/evaluated_json_ocr`

Placeholder route.

### Success Response

```text
json_ocr evaluated successfully
```

### Example

```bash
curl -X POST http://127.0.0.1:5000/evaluation_engine/evaluated_json_ocr
```

---

# 3. Fastify SaaS Backend APIs

Base URL: `http://127.0.0.1:3000`

Implemented in `gyaanverse-software`.

Important: the main Fastify app currently registers only `/health`. Module routes such as tenant routes are defined in files, but are not currently registered in `src/app.ts`.

## GET `/health`

Health check route.

### Success Response

```json
{
  "status": "ok"
}
```

### Example

```bash
curl http://127.0.0.1:3000/health
```

---

## Defined But Not Currently Exposed: Tenant Routes

These routes exist in `src/modules/tenant/tenant.routes.ts`, but are not active until `tenantRoutes` is registered in `src/app.ts`.

### POST `/tenants`

Creates a tenant.

Requires `req.user.id` to be populated by authentication middleware. Authentication middleware is not wired in the current app.

#### Request Body

```json
{
  "slug": "school-demo",
  "name": "School Demo"
}
```

#### Success Response

```json
{
  "id": "uuid",
  "slug": "school-demo",
  "name": "School Demo",
  "logoUrl": null,
  "ownerId": "uuid",
  "plan": "free",
  "status": "active",
  "createdAt": "2026-05-18T00:00:00.000Z",
  "updatedAt": "2026-05-18T00:00:00.000Z"
}
```

### GET `/tenants/:id`

Fetches a tenant by ID.

#### Success Response

```json
{
  "id": "uuid",
  "slug": "school-demo",
  "name": "School Demo",
  "logoUrl": null,
  "ownerId": "uuid",
  "plan": "free",
  "status": "active",
  "createdAt": "2026-05-18T00:00:00.000Z",
  "updatedAt": "2026-05-18T00:00:00.000Z"
}
```

#### Not Found Response

```json
{
  "error": "NOT_FOUND",
  "message": "Tenant not found"
}
```

### PATCH `/tenants/settings`

Updates settings for the current tenant.

Requires `req.tenant.id` to be populated by tenant middleware. Tenant middleware is not wired in the current app.

#### Request Body

```json
{
  "allowPublicMocks": true,
  "customDomain": "school.example.com"
}
```

#### Success Response

```json
{
  "success": true
}
```

---

## Scaffolded Modules With No Implemented Routes

The following route modules exist, but currently contain TODO placeholders only:

| Module | Route File |
| --- | --- |
| Auth | `src/modules/auth/auth.routes.ts` |
| Exam | `src/modules/exam/exam.routes.ts` |
| Exam Session | `src/modules/exam-session/exam-session.routes.ts` |
| Evaluation | `src/modules/evaluation/evaluation.routes.ts` |
| Report | `src/modules/report/report.routes.ts` |
| Storage | `src/modules/storage/storage.routes.ts` |
| Admin | `src/modules/admin/admin.routes.ts` |
| Analytics | `src/modules/analytics/analytics.routes.ts` |
| Notification | `src/modules/notification/notification.routes.ts` |
| Membership | `src/modules/membership/membership.routes.ts` |
| Billing | `src/modules/billing/billing.routes.ts` |
| Payment | `src/modules/payment/payment.routes.ts` |
| Class | `src/modules/class/class.routes.ts` |

---

# 4. Environment Variables

## Flask AI Engine

Common variables used by `engines`:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Required for OCR, evaluation, and embeddings. |
| `OPENAI_OCR_MODEL` | OCR model. Defaults to `gpt-4.1-mini`. |
| `OPENAI_CHAT_MODEL` | Evaluation model. Defaults to `gpt-5.4`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model. Defaults to `text-embedding-3-large`. |
| `OPENAI_TEMPERATURE` | Temperature setting. Defaults to `0`. |
| `OPENAI_REASONING_EFFORT` | Reasoning effort for RAG/document calls. Defaults to `high`. |
| `OPENAI_DIRECT_REASONING_EFFORT` | Reasoning effort for direct OCR evaluation. Defaults to `low`. |
| `OPENAI_TEXT_VERBOSITY` | Response verbosity. Defaults to `low`. |
| `QDRANT_COLLECTION_NAME` | Default Qdrant collection name. |
| `QDRANT_URL` | Qdrant URL. Defaults to `http://localhost:6333`. |
| `QDRANT_API_KEY` | Optional Qdrant API key. |
| `EMBEDDING_VECTOR_SIZE` | Embedding vector size. Defaults to `3072`. |

## SvelteKit App

| Variable | Description |
| --- | --- |
| `BACKEND_BASE_URL` | Flask engine URL. Defaults to `http://127.0.0.1:5000`. |
| `ANALYSIS_COLLECTION_NAME` | Optional collection name forwarded to Flask. |
| `EVALUATION_ENGINE_PATH` | Optional analysis route override. Defaults to `/evaluation_engine/get_analysis`. |

## Fastify SaaS Backend

The Fastify backend requires many variables at startup, including:

```text
DATABASE_URL
REDIS_URL
BETTER_AUTH_SECRET
BETTER_AUTH_URL
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET_NAME
R2_PUBLIC_URL
RESEND_API_KEY
MSG91_AUTH_KEY
MSG91_TEMPLATE_ID
RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET
RAZORPAY_WEBHOOK_SECRET
APP_DOMAIN
```

---

# 5. Notes And Gaps

- `/evaluation_engine/get_analysis` is RAG-backed and falls back to question-and-solution evaluation when retrieval is unavailable or below threshold.
- `/evaluation_engine/checked_json_ocr` intentionally remains the direct, non-RAG endpoint; use `/evaluation_engine/checked_json_ocr_with_rag` when grounding is required.
- The Fastify backend is a scaffold. Only `/health` is exposed by the app today.
- Expensive OCR/evaluation routes currently have no authentication or rate limiting at the Flask layer.
- Flask limits request bodies to 20 MiB by default; set `EVALUATION_MAX_REQUEST_BYTES` to tune the limit.
