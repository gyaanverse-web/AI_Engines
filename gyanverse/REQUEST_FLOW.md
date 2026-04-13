# Request Flow

This project has one main path for image analysis.

## Simple Overview

1. The user uploads an answer image in the Svelte frontend.
2. The frontend sends a `POST` request to `/api/analyze-image`.
3. The SvelteKit route forwards the request to the Flask backend at `/get_analysis`.
4. The Flask backend calls the shared `analyzer` function.
5. `analyzer` runs OCR if needed, then evaluates the steps.
6. The frontend renders the analysis output on the page.

## API Calls

### 1) Frontend to SvelteKit API

`POST /api/analyze-image`

Payload sent from the browser:

- `solution_url`: the uploaded image as a `data:image/...` URL
- `question`: the question shown to the user
- `top_k`: optional tuning value for analysis

### 2) SvelteKit API to Flask backend

`POST /get_analysis`

The SvelteKit endpoint acts as a proxy.
It forwards the image and question to the Python backend orchestrator.

### 3) Flask backend OCR step

If OCR data is not already provided, the Flask route calls:

`get_json_ocr(source)`

Inside that function:

- the image is sent to OpenAI Vision OCR
- the output is normalized into JSON steps
- each step keeps the original writing as closely as possible

### 4) Flask backend evaluation step

After OCR is ready, the backend calls:

`analyzer(...)`

Inside `analyzer(...)`, the backend may call:

`evaluate_ocr_steps_with_rag(...)`

This step compares the extracted steps against the expected solution logic and returns the analysis response.

## What Returns Back

The final response contains step-by-step analysis data such as:

- step status
- issue description
- step understanding
- step weight
- OCR text for each step

The frontend uses that response to show the analysis result.

## In Short

`User uploads image` -> `POST /api/analyze-image` -> `POST /checked_json_ocr` -> `OCR extraction` -> `step evaluation` -> `render result`

## Useful Backend Routes

- `/get_json_ocr` - OCR only
- `/get_analysis` - direct analysis route
