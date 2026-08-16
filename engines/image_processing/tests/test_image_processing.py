import base64
import unittest

import cv2
import numpy as np

from image_processing import create_app
from image_processing.modules.text_detector import (
    _validated_image_bytes,
    detect_text,
    detect_text_in_bytes,
)


def _png_bytes(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Unable to encode test image")
    return buffer.tobytes()


def _blank_page() -> np.ndarray:
    return np.full((600, 900, 3), 255, dtype=np.uint8)


def _text_page() -> np.ndarray:
    image = _blank_page()
    cv2.putText(
        image,
        "This page contains text 123",
        (70, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "x + y = 42",
        (70, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image


class ImageInputValidationTestCase(unittest.TestCase):
    def test_valid_image_bytes_are_accepted(self):
        image_bytes = _png_bytes(_blank_page())

        self.assertIs(_validated_image_bytes(image_bytes, "image/png"), image_bytes)

    def test_invalid_image_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid JPEG"):
            _validated_image_bytes(b"not an image", "image/png")

    def test_declared_mime_type_must_match_image_content(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validated_image_bytes(_png_bytes(_blank_page()), "image/jpeg")


class LocalTextDetectionTestCase(unittest.TestCase):
    def test_blank_page_does_not_contain_text(self):
        self.assertFalse(detect_text_in_bytes(_png_bytes(_blank_page()), "image/png"))

    def test_printed_text_and_math_are_detected(self):
        self.assertTrue(detect_text_in_bytes(_png_bytes(_text_page()), "image/png"))

    def test_simple_non_text_shapes_are_not_detected_as_text(self):
        image = _blank_page()
        cv2.rectangle(image, (100, 100), (400, 350), (0, 0, 0), 5)
        cv2.circle(image, (650, 220), 100, (0, 0, 0), 5)
        cv2.line(image, (100, 500), (800, 500), (0, 0, 0), 5)

        self.assertFalse(detect_text_in_bytes(_png_bytes(image), "image/png"))

    def test_aligned_dots_are_not_detected_as_text(self):
        image = cv2.cvtColor(_blank_page(), cv2.COLOR_BGR2GRAY)
        for x_position in range(50, 850, 40):
            cv2.circle(image, (x_position, 300), 5, 0, -1)

        self.assertFalse(detect_text_in_bytes(_png_bytes(image), "image/png"))

    def test_dense_visual_noise_is_not_detected_as_text(self):
        noise = np.random.default_rng(7).integers(
            0,
            256,
            (600, 900),
            dtype=np.uint8,
        )

        self.assertFalse(detect_text_in_bytes(_png_bytes(noise), "image/png"))

    def test_data_url_is_processed_locally(self):
        encoded = base64.b64encode(_png_bytes(_text_page())).decode("ascii")

        self.assertTrue(detect_text(f"data:image/png;base64,{encoded}"))


class ImageProcessingRouteTestCase(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_image_source_returns_presence_decision(self):
        encoded = base64.b64encode(_png_bytes(_text_page())).decode("ascii")
        response = self.client.post(
            "/image_processing/contains_text",
            json={"image_source": f"data:image/png;base64,{encoded}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"contains_text": True})

    def test_old_source_field_is_rejected(self):
        response = self.client.post(
            "/image_processing/contains_text",
            json={"source": "https://example.com/page.png"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "image_source must be a non-empty string"},
        )

    def test_multipart_upload_is_rejected(self):
        response = self.client.post(
            "/image_processing/contains_text",
            data={"image": "not-used"},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Request body must be a JSON object containing image_source"},
        )

    def test_missing_image_source_returns_validation_error(self):
        response = self.client.post("/image_processing/contains_text", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "image_source must be a non-empty string"},
        )


if __name__ == "__main__":
    unittest.main()
