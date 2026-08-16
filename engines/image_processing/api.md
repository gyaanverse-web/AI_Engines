# Image Processing API

The combined service mounts this engine at `/image_processing`.

Run it with the other engines from `engines/`:

```bash
PYTHONPATH=. .venv/bin/python app.py
```

It can also run by itself:

```bash
PYTHONPATH=. .venv/bin/python image_processing/app.py
```

## Check whether an image contains text

```http
POST /image_processing/contains_text
Content-Type: application/json
```

Send an HTTPS URL, base64 image data URL, or allowed server-local path in the JSON
request body. `image_source` is the only image input field:

```bash
curl -X POST http://localhost:5000/image_processing/contains_text \
  -H "Content-Type: application/json" \
  -d '{"image_source":"https://example.com/page.jpg"}'
```

The endpoint deliberately returns only the presence decision:

```json
{
  "contains_text": true
}
```

Detection runs locally with OpenCV. It does not perform OCR, extract or return text, use
an LLM, require an API key, or call an inference service. When an HTTP source is supplied,
the service only downloads the image; data URLs and local paths require no network access.

Supported images are JPEG, PNG, WEBP, and GIF. Local paths must resolve inside
`IMAGE_PROCESSING_LOCAL_FILE_ROOT`. Uploads and local/data-URL images must stay within
`IMAGE_PROCESSING_MAX_IMAGE_BYTES`.
