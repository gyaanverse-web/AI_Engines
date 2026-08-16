import unittest
from os import environ
from unittest.mock import patch


environ.setdefault("OPENAI_API_KEY", "test-openai-key")
environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from question_analysis.categories import CATEGORY_KEYS
from evaluation_engine import create_app
from evaluation_engine.modules.analysis_engine import analyze_solution
from evaluation_engine.modules.provider_engine import (
    _apply_deterministic_step_checks,
    _safe_eval_numeric,
    chunk_document_text,
    extract_text_file_content,
)
from evaluation_engine.modules.ocr_engine import _build_image_input
from evaluation_engine.modules.testing_engine import (
    _apply_context_unit_guard,
    _build_blocks_from_steps,
    _build_response_payload,
    _build_summary,
    _finalize_step_result,
    _get_rag_context,
    _reconstruct_solution_steps,
    evaluate_ocr_steps_with_rag,
)


class EvaluationHardeningTestCase(unittest.TestCase):
    def test_reconstruction_uses_model_grouping_but_never_model_text(self):
        ocr_steps = [
            {"stepId": "1", "text": "7x - 2 = 22x + 10"},
            {"stepId": "2", "text": "-15x = 12"},
            {"stepId": "3", "text": "x = -4/5"},
        ]
        model_response = {
            "steps": [
                {
                    "stepId": "rewritten",
                    "sourceStepIds": ["1", "2"],
                    "text": "x = 999 (invented)",
                }
            ]
        }

        with patch(
            "evaluation_engine.modules.testing_engine._generate_json_response",
            return_value=model_response,
        ):
            result = _reconstruct_solution_steps(ocr_steps, "Solve for x")

        self.assertEqual(result[0]["stepId"], "1")
        self.assertEqual(result[0]["sourceStepIds"], ["1", "2"])
        self.assertEqual(result[0]["text"], "7x - 2 = 22x + 10 -15x = 12")
        self.assertEqual(result[1]["text"], "x = -4/5")
        self.assertNotIn("999", str(result))

    def test_reconstruction_failure_preserves_all_original_steps(self):
        ocr_steps = [
            {"stepId": "1", "text": "Given m = 2 kg"},
            {"stepId": "2", "text": "F = ma"},
        ]
        with patch(
            "evaluation_engine.modules.testing_engine._generate_json_response",
            side_effect=RuntimeError("provider unavailable"),
        ):
            result = _reconstruct_solution_steps(ocr_steps, "Find force")

        self.assertEqual([step["text"] for step in result], ["Given m = 2 kg", "F = ma"])

    def test_unlabeled_solution_keeps_one_block_per_logical_step(self):
        steps = [
            {"stepId": "1", "sourceStepIds": ["1"], "text": "F = ma"},
            {"stepId": "2", "sourceStepIds": ["2"], "text": "F = 10 N"},
        ]

        annotated, blocks = _build_blocks_from_steps(steps, [])

        self.assertEqual([step["blockId"] for step in annotated], ["block_1", "block_2"])
        self.assertEqual([block["stepIds"] for block in blocks], [["1"], ["2"]])

    def test_finalization_does_not_accept_model_rewrite_of_student_text(self):
        scores = {category: 0 for category in CATEGORY_KEYS}
        profile = {
            "scores": scores,
            "weights": scores,
            "primary_category": "concept_based",
            "secondary_categories": [],
        }
        step = {
            "stepId": "1",
            "sourceStepIds": ["1"],
            "blockId": "block_1",
            "blockLabel": "block_1",
            "question_part_label": "",
            "question_part_text": "",
            "text": "2 + 2 = 5",
        }
        raw_result = {
            "text": "2 + 2 = 4",
            "step_status": "right",
            "step_weight": 0.8,
            "step_type": "calculation_based",
            "topic": "Arithmetic",
            "step_understanding": "The student adds two numbers.",
            "description": "",
        }

        result = _finalize_step_result(
            question="Calculate 2 + 2",
            question_profile=profile,
            step=step,
            raw_result=raw_result,
        )

        self.assertEqual(result["text"], "2 + 2 = 5")

    def test_response_payload_uses_flat_versioned_contract(self):
        step_result = {
            "stepId": "1",
            "sourceStepIds": ["1"],
            "blockId": "block_1",
            "blockLabel": "block_1",
            "question_part_label": "",
            "question_part_text": "",
            "text": "x = 2",
            "step_status": "right",
            "step_weight": 1,
            "step_type": "calculation_based",
            "topic": "Linear Equations",
            "step_understanding": "The student solves for x.",
            "description": "",
        }

        payload = _build_response_payload(
            final_results=[step_result],
            full_marks=5,
            grounding_status="not_requested",
        )

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(
            list(payload),
            ["schema_version", "steps", "summary", "grounding"],
        )
        self.assertEqual(payload["summary"]["obtained_marks"], 5)
        self.assertEqual(payload["summary"]["percentage"], 100)
        self.assertEqual(payload["summary"]["scored_step_count"], 1)
        self.assertEqual(len(payload["steps"]), 1)
        self.assertEqual(payload["grounding"]["status"], "not_requested")
        self.assertNotIn("blocks", payload)
        self.assertNotIn("response", payload)
        self.assertNotIn("sourceStepIds", payload["steps"][0])

    def test_unscored_steps_are_reported_but_do_not_change_score(self):
        steps = [
            {
                "stepId": "1",
                "step_status": "right",
                "step_weight": 0.3,
                "counts_toward_score": True,
            },
            {
                "stepId": "2",
                "step_status": "wrong",
                "step_weight": 0,
                "counts_toward_score": False,
            },
        ]

        summary = _build_summary(steps, full_marks=5)

        self.assertEqual(summary["overall_status"], "right")
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["scored_step_count"], 1)
        self.assertEqual(summary["status_breakdown"]["wrong"], 1)
        self.assertEqual(summary["percentage"], 100)
        self.assertEqual(summary["obtained_marks"], 5)

    def test_rag_context_filters_weak_chunks_and_returns_provenance(self):
        chunks = [
            {
                "score": 0.91,
                "document_id": "science-10",
                "chunk_index": 4,
                "text": "Force equals mass times acceleration.",
                "metadata": {"chapter": "Force"},
            },
            {
                "score": 0.1,
                "document_id": "irrelevant",
                "chunk_index": 2,
                "text": "This weak chunk must not be supplied.",
                "metadata": {},
            },
        ]
        with patch(
            "evaluation_engine.modules.testing_engine.retrieve_relevant_chunks",
            return_value=chunks,
        ):
            context, reason, sources = _get_rag_context(
                question="State Newton's second law",
                steps=[{"stepId": "1", "text": "F = ma"}],
                collection_name="science",
                top_k=5,
            )

        self.assertIsNone(reason)
        self.assertIn("[Source 1", context)
        self.assertNotIn("weak chunk", context)
        self.assertEqual(sources[0]["document_id"], "science-10")

    def test_fractional_assignment_is_evaluated_as_a_fraction(self):
        result = _apply_deterministic_step_checks(
            step_result={"step_status": "unknown", "description": ""},
            question="7x - 2 = 2(11x + 5). Find x.",
            step_text="x = -4/5",
            local_context="1: 7x - 2 = 2(11x + 5)",
        )

        self.assertEqual(result["step_status"], "right")
        self.assertEqual(result["description"], "")

    def test_force_unit_is_not_confused_with_acceleration_in_multipart_question(self):
        result = _apply_context_unit_guard(
            step_result={
                "step_status": "right",
                "description": "",
                "step_weight": 1,
            },
            question="Find (a) acceleration and (b) force on wagon 2.",
            step_text="F = 20520 \\mathrm{N}",
            local_context="5: (b) Force on wagon 2",
        )

        self.assertEqual(result["step_status"], "right")

    def test_numeric_evaluator_rejects_code_and_unsafe_exponents(self):
        self.assertEqual(_safe_eval_numeric("(2 + 3) * 4"), 20)
        self.assertIsNone(_safe_eval_numeric("__import__('os')"))
        self.assertIsNone(_safe_eval_numeric("9^999999"))

    def test_chunker_rejects_non_progressing_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document_text("content", chunk_size=100, overlap=100)

    def test_rag_file_reader_rejects_paths_outside_document_root(self):
        with self.assertRaisesRegex(PermissionError, "RAG_DOCUMENT_ROOT"):
            extract_text_file_content("/etc/passwd")

    def test_ocr_rejects_invalid_or_unsupported_data_urls(self):
        with self.assertRaisesRegex(ValueError, "base64"):
            _build_image_input("data:image/png;base64,not-valid-base64")
        with self.assertRaisesRegex(ValueError, "MIME"):
            _build_image_input("data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=")

    def test_analyzer_uses_rag_by_default(self):
        expected = {
            "schema_version": "1.0",
            "steps": [],
            "summary": {},
            "grounding": {"status": "used"},
        }
        with patch(
            "evaluation_engine.modules.testing_engine.evaluate_ocr_steps_with_rag",
            return_value=expected,
        ) as rag_evaluator:
            result = analyze_solution(
                ocr_data=[{"stepId": "1", "text": "F = ma"}],
                question="State Newton's second law",
            )

        self.assertEqual(result, expected)
        rag_evaluator.assert_called_once()

    def test_rag_evaluator_rejects_invalid_top_k_before_provider_calls(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            evaluate_ocr_steps_with_rag(
                ocr_data=[{"stepId": "1", "text": "F = ma"}],
                question="State Newton's second law",
                top_k=0,
            )

    def test_complete_rag_flow_returns_grounded_step_details_and_summary(self):
        def fake_generation(*, system_instruction, **_kwargs):
            if "reconstruct student solution steps" in system_instruction:
                return {
                    "steps": [
                        {"stepId": "1", "sourceStepIds": ["1"], "text": "rewritten"},
                        {"stepId": "2", "sourceStepIds": ["2"], "text": "rewritten"},
                    ]
                }
            if "prepare context" in system_instruction:
                return {
                    "goal": "Explain photosynthesis",
                    "method_outline": ["Identify required inputs"],
                    "carry_forward_policy": "Evaluate each statement",
                    "global_notes": [],
                    "primary_category": "concept_based",
                }
            return {
                "response": [
                    {
                        "stepId": "1",
                        "text": "model rewrite",
                        "step_status": "right",
                        "step_weight": 0.4,
                        "step_type": "concept_based",
                        "topic": "Photosynthesis",
                        "step_understanding": "The student identifies sunlight as an input.",
                        "description": "",
                    },
                    {
                        "stepId": "2",
                        "text": "model rewrite",
                        "step_status": "right",
                        "step_weight": 0.6,
                        "step_type": "concept_based",
                        "topic": "Photosynthesis",
                        "step_understanding": "The student identifies carbon dioxide as an input.",
                        "description": "",
                    },
                ]
            }

        retrieved = [
            {
                "score": 0.9,
                "document_id": "science-book",
                "chunk_index": 1,
                "text": "Photosynthesis uses sunlight and carbon dioxide.",
                "metadata": {"chapter": "Life Processes"},
            }
        ]
        with (
            patch(
                "evaluation_engine.modules.testing_engine._generate_json_response",
                side_effect=fake_generation,
            ),
            patch(
                "evaluation_engine.modules.testing_engine.retrieve_relevant_chunks",
                return_value=retrieved,
            ),
        ):
            payload = evaluate_ocr_steps_with_rag(
                ocr_data=[
                    {"stepId": "1", "text": "Sunlight is required."},
                    {"stepId": "2", "text": "Carbon dioxide is required."},
                ],
                question="Name two inputs needed for photosynthesis.",
                collection_name="science",
                top_k=3,
                full_marks=4,
            )

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["grounding"]["status"], "used")
        self.assertEqual(
            payload["grounding"]["sources"][0]["document_id"],
            "science-book",
        )
        self.assertEqual([step["text"] for step in payload["steps"]], [
            "Sunlight is required.",
            "Carbon dioxide is required.",
        ])
        self.assertEqual(len(payload["steps"]), 2)
        self.assertTrue(all(step["counts_toward_score"] for step in payload["steps"]))
        self.assertEqual(payload["summary"]["obtained_marks"], 4)


class EvaluationRouteValidationTestCase(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_non_object_json_is_rejected(self):
        response = self.client.post("/evaluation_engine/get_analysis", json=[])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Request body must be a JSON object")

    def test_invalid_top_k_is_rejected(self):
        response = self.client.post(
            "/evaluation_engine/get_analysis",
            json={"image_source": "data:image/png;base64,AA==", "top_k": 21},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("top_k", response.get_json()["error"])

    def test_invalid_collection_name_is_rejected_before_ocr(self):
        response = self.client.post(
            "/evaluation_engine/get_analysis",
            json={
                "image_source": "data:image/png;base64,AA==",
                "collection_name": "../../secrets",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("collection_name", response.get_json()["error"])

    def test_malformed_ocr_step_is_rejected(self):
        response = self.client.post(
            "/evaluation_engine/checked_json_ocr",
            json={"ocr_data": ["not-an-object"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("step 1", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
