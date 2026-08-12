import unittest
from os import environ

environ.setdefault("OPENAI_API_KEY", "test-openai-key")
environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from evaluation_engine.modules.testing_engine import (
    _build_block_results,
    _build_blocks_from_steps,
    _build_public_response,
    _coerce_description,
    _coerce_step_understanding,
    _extract_question_parts,
    _pick_block_understanding,
)
from evaluation_engine.modules.provider_engine import (
    _apply_deterministic_step_checks,
)


class BlockEvaluationTestCase(unittest.TestCase):
    def test_wrong_integer_answer_does_not_round_fractional_correction(self):
        result = _apply_deterministic_step_checks(
            step_result={"step_status": "wrong", "description": ""},
            question="7x - 2 = 2(11x+5). Find the value of x.",
            step_text=r"\therefore x = -1",
            local_context="1: 7x - 2 = 2(11x+5)",
        )

        self.assertEqual(result["step_status"], "wrong")
        self.assertIn(r"x = -0.8", result["description"])
        self.assertNotIn(r"x = -1", result["description"])

    def test_description_keeps_complete_correction(self):
        description = (
            "Error: Incorrect calculation of the equation; Correct step: "
            "7x - 22x = 10 + 2, which leads to -15x = 12 and therefore x = -4/5"
        )

        result = _coerce_description("wrong", description, "calculation_based")

        self.assertEqual(result, description)
        self.assertFalse(result.endswith("..."))

    def test_block_understanding_keeps_complete_sentence(self):
        understanding = (
            "The student correctly identified the equation to be solved and expanded "
            "the right side before attempting to isolate the variable."
        )

        finalized_understanding = _coerce_step_understanding(understanding)
        result = _pick_block_understanding(
            [{"step_understanding": finalized_understanding}]
        )

        self.assertEqual(result, understanding)
        self.assertFalse(result.endswith("..."))

    def test_extract_question_parts_splits_labeled_subquestions(self):
        question = (
            "Calculate: (a) the net accelerating force, "
            "(b) the acceleration of the train, and "
            "(c) the force of wagon 1 on wagon 2."
        )

        parts = _extract_question_parts(question)

        self.assertEqual(
            parts,
            [
                {"label": "a", "text": "the net accelerating force"},
                {"label": "b", "text": "the acceleration of the train"},
                {"label": "c", "text": "the force of wagon 1 on wagon 2"},
            ],
        )

    def test_build_blocks_from_steps_groups_by_part_labels(self):
        question_parts = [
            {"label": "a", "text": "the net accelerating force"},
            {"label": "b", "text": "the acceleration of the train"},
            {"label": "c", "text": "the force of wagon 1 on wagon 2"},
        ]
        steps = [
            {"stepId": "1", "sourceStepIds": ["1"], "text": "(a) Net accelerating force:"},
            {"stepId": "2", "sourceStepIds": ["2"], "text": "F_{net} = 40000 - 5000 = 35000"},
            {"stepId": "3", "sourceStepIds": ["3"], "text": "(b) Acceleration of the train:"},
            {"stepId": "4", "sourceStepIds": ["4"], "text": "a = 35000 / 18000"},
            {"stepId": "5", "sourceStepIds": ["5"], "text": "(c) Force of wagon 1 on wagon 2:"},
            {"stepId": "6", "sourceStepIds": ["6"], "text": "F = 20520 N"},
        ]

        annotated_steps, blocks = _build_blocks_from_steps(steps, question_parts)

        self.assertEqual([block["blockLabel"] for block in blocks], ["a", "b", "c"])
        self.assertEqual([block["stepIds"] for block in blocks], [["1", "2"], ["3", "4"], ["5", "6"]])
        self.assertEqual(annotated_steps[1]["question_part_text"], "the net accelerating force")
        self.assertEqual(annotated_steps[3]["question_part_text"], "the acceleration of the train")
        self.assertEqual(annotated_steps[5]["question_part_text"], "the force of wagon 1 on wagon 2")

    def test_build_block_results_calculates_marks_from_step_weights(self):
        step_results = [
            {
                "stepId": "1",
                "blockId": "block_1",
                "blockLabel": "a",
                "question_part_label": "a",
                "question_part_text": "the net accelerating force",
                "text": "F_{net} = 35000",
                "step_status": "right",
                "step_weight": 0.4,
                "step_type": "calculation_based",
                "topic": "Net Force",
                "step_understanding": "Computes the net force.",
                "description": "",
                "sourceStepIds": ["1"],
            },
            {
                "stepId": "2",
                "blockId": "block_2",
                "blockLabel": "b",
                "question_part_label": "b",
                "question_part_text": "the acceleration of the train",
                "text": "a = 1.94",
                "step_status": "incomplete",
                "step_weight": 0.3,
                "step_type": "calculation_based",
                "topic": "Acceleration",
                "step_understanding": "Computes the acceleration but misses the unit.",
                "description": "Missing: unit; Correct step: write a = 1.94 m/s^2",
                "sourceStepIds": ["2"],
            },
            {
                "stepId": "3",
                "blockId": "block_3",
                "blockLabel": "c",
                "question_part_label": "c",
                "question_part_text": "the force of wagon 1 on wagon 2",
                "text": "F = 20520 N",
                "step_status": "wrong",
                "step_weight": 0.3,
                "step_type": "reasoning_based",
                "topic": "Coupling Force",
                "step_understanding": "Uses the final block force relation.",
                "description": "Error: uses the wrong friction assumption; Correct step: isolate the force on wagons 2 to 5.",
                "sourceStepIds": ["3"],
            },
        ]

        blocks, summary = _build_block_results(step_results, full_marks=5)

        self.assertEqual(summary["full_marks"], 5)
        self.assertEqual(summary["obtained_marks"], 2.75)
        self.assertEqual(summary["percentage"], 55.0)
        self.assertEqual([block["obtained_marks"] for block in blocks], [2, 0.75, 0])

    def test_build_public_response_matches_exp03_shape_with_blocks_as_steps(self):
        step_results = [
            {
                "stepId": "1",
                "blockId": "block_1",
                "blockLabel": "a",
                "question_part_label": "a",
                "question_part_text": "the net accelerating force",
                "text": "(a) Net force = 35000 N",
                "step_status": "right",
                "step_weight": 0.4,
                "step_type": "calculation_based",
                "topic": "Net Force",
                "step_understanding": "Computes the net force.",
                "description": "",
                "sourceStepIds": ["1"],
            },
            {
                "stepId": "2",
                "blockId": "block_2",
                "blockLabel": "b",
                "question_part_label": "b",
                "question_part_text": "the acceleration of the train",
                "text": "(b) a = 1.94",
                "step_status": "incomplete",
                "step_weight": 0.6,
                "step_type": "calculation_based",
                "topic": "Acceleration",
                "step_understanding": "Computes the acceleration but misses the unit.",
                "description": "Missing: unit; Correct step: write a = 1.94 m/s^2",
                "sourceStepIds": ["2"],
            },
        ]

        blocks, summary = _build_block_results(step_results, full_marks=5)
        response = _build_public_response(blocks, float(summary["total_weight"]))

        self.assertEqual(
            list(response[0].keys()),
            [
                "stepId",
                "text",
                "step_status",
                "step_weight",
                "topic",
                "step_understanding",
                "description",
            ],
        )
        self.assertEqual(response[0]["stepId"], "1")
        self.assertEqual(response[0]["step_status"], "right")
        self.assertEqual(response[0]["step_weight"], 0.4)
        self.assertEqual(response[1]["step_status"], "incomplete")
        self.assertEqual(response[1]["description"], "Missing: unit; Correct step: write a = 1.94 m/s^2")


if __name__ == "__main__":
    unittest.main()
