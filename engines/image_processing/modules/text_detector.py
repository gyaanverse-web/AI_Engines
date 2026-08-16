import base64
import binascii
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np


IMAGE_PROCESSING_MAX_IMAGE_BYTES = int(
    os.getenv("IMAGE_PROCESSING_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
)
IMAGE_PROCESSING_MAX_PIXELS = int(
    os.getenv("IMAGE_PROCESSING_MAX_PIXELS", str(25_000_000))
)
IMAGE_PROCESSING_URL_TIMEOUT = float(
    os.getenv("IMAGE_PROCESSING_URL_TIMEOUT", "10")
)
TEXT_DETECTION_MAX_DIMENSION = int(
    os.getenv("TEXT_DETECTION_MAX_DIMENSION", "1600")
)
TEXT_DETECTION_MIN_COMPONENTS = int(
    os.getenv("TEXT_DETECTION_MIN_COMPONENTS", "3")
)
ENGINE_ROOT = Path(__file__).resolve().parents[2]
_configured_local_root = Path(
    os.getenv("IMAGE_PROCESSING_LOCAL_FILE_ROOT", "..")
).expanduser()
IMAGE_PROCESSING_LOCAL_FILE_ROOT = (
    _configured_local_root
    if _configured_local_root.is_absolute()
    else ENGINE_ROOT / _configured_local_root
).resolve()
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _detected_mime_type(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _validated_image_bytes(
    image_bytes: bytes,
    declared_mime_type: str | None = None,
) -> bytes:
    if not image_bytes:
        raise ValueError("Image is empty")
    if len(image_bytes) > IMAGE_PROCESSING_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image must not exceed {IMAGE_PROCESSING_MAX_IMAGE_BYTES} bytes"
        )

    detected_mime_type = _detected_mime_type(image_bytes)
    if detected_mime_type is None:
        raise ValueError("Image must be a valid JPEG, PNG, WEBP, or GIF")

    normalized_declared_type = (declared_mime_type or "").lower().split(";", 1)[0]
    if normalized_declared_type not in {"", "application/octet-stream"}:
        if normalized_declared_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {normalized_declared_type}")
        if normalized_declared_type != detected_mime_type:
            raise ValueError("Image content does not match its declared MIME type")
    return image_bytes


def _read_remote_image(source: str) -> tuple[bytes, str | None]:
    request = Request(source, headers={"User-Agent": "EducationalAI-image-detector/1.0"})
    try:
        with urlopen(request, timeout=IMAGE_PROCESSING_URL_TIMEOUT) as response:
            declared_size = response.headers.get("Content-Length")
            if declared_size:
                try:
                    parsed_size = int(declared_size)
                except (TypeError, ValueError):
                    parsed_size = None
                if (
                    parsed_size is not None
                    and parsed_size > IMAGE_PROCESSING_MAX_IMAGE_BYTES
                ):
                    raise ValueError(
                        f"Image must not exceed {IMAGE_PROCESSING_MAX_IMAGE_BYTES} bytes"
                    )
            image_bytes = response.read(IMAGE_PROCESSING_MAX_IMAGE_BYTES + 1)
            mime_type = response.headers.get_content_type()
    except ValueError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ValueError("Unable to download image source") from exc

    return image_bytes, mime_type


def _read_image_source(source: str) -> tuple[bytes, str | None]:
    if source.startswith("data:image/"):
        header, separator, encoded = source.partition(",")
        mime_type = header.removeprefix("data:").split(";", 1)[0].lower()
        if not separator or ";base64" not in header.lower():
            raise ValueError("Image data URL must contain base64-encoded data")
        if mime_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {mime_type}")
        try:
            return base64.b64decode(encoded, validate=True), mime_type
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image data URL contains invalid base64 data") from exc

    if source.startswith(("http://", "https://")):
        return _read_remote_image(source)

    image_path = Path(source).expanduser()
    if not image_path.is_absolute():
        image_path = IMAGE_PROCESSING_LOCAL_FILE_ROOT / image_path
    image_path = image_path.resolve()
    try:
        image_path.relative_to(IMAGE_PROCESSING_LOCAL_FILE_ROOT)
    except ValueError as exc:
        raise PermissionError(
            "Image path must be inside IMAGE_PROCESSING_LOCAL_FILE_ROOT: "
            f"{IMAGE_PROCESSING_LOCAL_FILE_ROOT}"
        ) from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {source}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError("Local image must be JPEG, PNG, WEBP, or GIF")
    if image_path.stat().st_size > IMAGE_PROCESSING_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image must not exceed {IMAGE_PROCESSING_MAX_IMAGE_BYTES} bytes"
        )
    return image_path.read_bytes(), mime_type


def _decode_grayscale_image(image_bytes: bytes) -> np.ndarray:
    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError("Image data could not be decoded")

    height, width = image.shape
    if height * width > IMAGE_PROCESSING_MAX_PIXELS:
        raise ValueError(f"Image must not exceed {IMAGE_PROCESSING_MAX_PIXELS} pixels")

    largest_dimension = max(height, width)
    if largest_dimension > TEXT_DETECTION_MAX_DIMENSION:
        scale = TEXT_DETECTION_MAX_DIMENSION / largest_dimension
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _component_rows(binary_image: np.ndarray) -> list[list[tuple[int, int, int, int, int]]]:
    image_height, image_width = binary_image.shape
    minimum_height = max(3, round(min(image_height, image_width) * 0.004))
    maximum_height = max(minimum_height + 1, round(image_height * 0.20))
    maximum_width = max(3, round(image_width * 0.30))

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8,
    )
    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not minimum_height <= height <= maximum_height:
            continue
        if not 1 <= width <= maximum_width:
            continue

        aspect_ratio = width / height
        fill_ratio = area / (width * height)
        if not 0.07 <= aspect_ratio <= 12:
            continue
        if not 0.04 <= fill_ratio <= 0.66:
            continue
        components.append((x, y, width, height, area))

    if len(components) < TEXT_DETECTION_MIN_COMPONENTS or len(components) > 6000:
        return []

    rows: list[list[tuple[int, int, int, int, int]]] = []
    row_centers: list[float] = []
    row_heights: list[float] = []
    for component in sorted(components, key=lambda item: (item[1] + item[3] / 2, item[0])):
        center_y = component[1] + component[3] / 2
        best_row = None
        best_distance = float("inf")
        for index, (row_center, row_height) in enumerate(
            zip(row_centers, row_heights)
        ):
            distance = abs(center_y - row_center)
            tolerance = max(minimum_height, 0.65 * max(component[3], row_height))
            height_ratio = component[3] / row_height
            if distance <= tolerance and 0.30 <= height_ratio <= 3.30:
                if distance < best_distance:
                    best_row = index
                    best_distance = distance

        if best_row is None:
            rows.append([component])
            row_centers.append(center_y)
            row_heights.append(float(component[3]))
            continue

        rows[best_row].append(component)
        centers = [item[1] + item[3] / 2 for item in rows[best_row]]
        heights = [item[3] for item in rows[best_row]]
        row_centers[best_row] = float(np.median(centers))
        row_heights[best_row] = float(np.median(heights))

    return rows


def _row_has_text_run(row: list[tuple[int, int, int, int, int]]) -> bool:
    if len(row) < TEXT_DETECTION_MIN_COMPONENTS:
        return False

    sorted_components = sorted(row, key=lambda item: item[0])
    typical_height = float(np.median([item[3] for item in sorted_components]))
    maximum_gap = max(4, round(typical_height * 3.0))
    run: list[tuple[int, int, int, int, int]] = []
    run_right = -1

    def is_text_like(components: list[tuple[int, int, int, int, int]]) -> bool:
        if len(components) < TEXT_DETECTION_MIN_COMPONENTS:
            return False
        left = min(item[0] for item in components)
        right = max(item[0] + item[2] for item in components)
        top = min(item[1] for item in components)
        bottom = max(item[1] + item[3] for item in components)
        run_area = max(1, (right - left) * (bottom - top))
        ink_ratio = sum(item[4] for item in components) / run_area
        return right - left >= typical_height * 1.5 and 0.025 <= ink_ratio <= 0.75

    for component in sorted_components:
        gap = component[0] - run_right
        if run and gap > maximum_gap:
            if is_text_like(run):
                return True
            run = []
        run.append(component)
        run_right = max(run_right, component[0] + component[2])

    return is_text_like(run)


def _binary_image_contains_text(binary_image: np.ndarray) -> bool:
    return any(_row_has_text_run(row) for row in _component_rows(binary_image))


def _image_contains_text(image: np.ndarray) -> bool:
    if image.shape[0] < 8 or image.shape[1] < 8 or float(image.std()) < 2.0:
        return False

    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    edge_density = float(np.count_nonzero(cv2.Canny(denoised, 80, 180))) / image.size
    if edge_density > 0.25:
        return False

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    _threshold, dark_foreground = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    if _binary_image_contains_text(dark_foreground):
        return True

    light_foreground = cv2.bitwise_not(dark_foreground)
    return _binary_image_contains_text(light_foreground)


def detect_text(source: str) -> bool:
    """Detect visible text locally in a URL, data URL, or allowed local image."""
    image_bytes, mime_type = _read_image_source(source)
    return detect_text_in_bytes(image_bytes, mime_type)


def detect_text_in_bytes(
    image_bytes: bytes,
    declared_mime_type: str | None = None,
) -> bool:
    """Detect visible text in uploaded bytes without OCR or external inference."""
    validated_bytes = _validated_image_bytes(image_bytes, declared_mime_type)
    return _image_contains_text(_decode_grayscale_image(validated_bytes))
