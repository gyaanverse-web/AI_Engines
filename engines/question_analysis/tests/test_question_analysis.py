import unittest
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation_engine import create_app
from question_analysis.categories import CATEGORY_KEYS
from question_analysis.ml_classifier import (
    MLClassifierService,
    build_ml_features,
)
from question_analysis.ml_training import (
    train_model_from_dataset,
)
from question_analysis.normalizer import normalize_scores
from question_analysis.question_skill_weightage import (
    analyze_question_skill_weightage,
)
from question_analysis.scorer import (
    calculate_question_skill_weightage,
)
from question_analysis.synthetic_dataset import (
    write_synthetic_dataset,
)


class QuestionAnalysisTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = create_app()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_validation_rejects_missing_question(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={"chapter": "Simple Interest"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "question is required"})

    def test_validation_rejects_missing_chapter(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={"question": "Calculate the answer"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "chapter is required"})

    def test_validation_rejects_non_boolean_use_ml(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={
                "question": "Calculate the answer",
                "chapter": "Simple Interest",
                "use_ml": "true",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "use_ml must be a boolean"})

    def test_score_normalization_total_equals_100(self):
        result = normalize_scores(
            {
                "concept_based": 11,
                "formula_based": 17,
                "calculation_based": 33,
                "reasoning_based": 9,
            }
        )

        self.assertEqual(sum(result.values()), 100)
        self.assertEqual(list(result.keys()), CATEGORY_KEYS)

    def test_simple_interest_question_returns_high_calculation_and_formula_scores(self):
        result = calculate_question_skill_weightage(
            question="calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
            chapter="simple interest",
        )

        self.assertGreaterEqual(result["calculation_based"], 40)
        self.assertGreaterEqual(result["formula_based"], 25)

    def test_photosynthesis_question_returns_high_concept_score(self):
        result = calculate_question_skill_weightage(
            question="explain the process of photosynthesis and its effect on plants.",
            chapter="photosynthesis",
        )

        self.assertGreaterEqual(result["concept_based"], 50)

    def test_prove_question_returns_high_proof_score(self):
        result = calculate_question_skill_weightage(
            question="prove that the sum of angles in a triangle is 180 degrees.",
            chapter="proof",
        )

        self.assertGreaterEqual(result["proof_or_derivation_based"], 40)

    def test_table_chart_question_boosts_data_interpretation(self):
        result = calculate_question_skill_weightage(
            question="study the following table and chart, then analyze the data given below.",
            chapter="data interpretation",
        )

        self.assertGreaterEqual(result["data_interpretation_based"], 35)

    def test_diagram_draw_question_boosts_diagram_category(self):
        result = calculate_question_skill_weightage(
            question="draw and label the diagram of a circuit shown in the figure.",
            chapter="physics",
        )

        self.assertGreaterEqual(result["diagram_based"], 40)

    def test_response_contains_exactly_10_category_keys(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={
                "question": "Calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
                "chapter": "Simple Interest",
            },
        )

        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(payload.keys()), CATEGORY_KEYS)
        self.assertEqual(len(payload), 10)

    def test_response_does_not_contain_extra_metadata_fields(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={
                "question": "Explain photosynthesis in detail.",
                "chapter": "Photosynthesis",
            },
        )

        payload = response.get_json()

        for forbidden_key in [
            "success",
            "message",
            "data",
            "confidence",
            "explanation",
            "evidence",
            "difficulty",
            "question_type",
        ]:
            self.assertNotIn(forbidden_key, payload)

    def test_response_header_indicates_rule_only_mode(self):
        response = self.client.post(
            "/api/v1/question-analysis/analyze",
            json={
                "question": "Explain photosynthesis in detail.",
                "chapter": "Photosynthesis",
                "use_ml": False,
            },
        )

        self.assertEqual(response.headers.get("X-Question-Analysis-Mode"), "rule_only")

    def test_orchestrator_returns_exact_category_set(self):
        payload = analyze_question_skill_weightage(
            {
                "question": "List the steps of photosynthesis.",
                "chapter": "Photosynthesis",
            }
        )

        self.assertEqual(list(payload.keys()), CATEGORY_KEYS)

    def test_synthetic_dataset_generation_writes_records(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            write_synthetic_dataset(path=dataset_path)

            self.assertTrue(dataset_path.exists())
            self.assertGreater(dataset_path.stat().st_size, 0)

    def test_ml_training_and_prediction_returns_all_categories(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            model_path = Path(temp_dir) / "model.json"
            write_synthetic_dataset(path=dataset_path)
            train_model_from_dataset(dataset_path=dataset_path, model_path=model_path)

            ml_service = MLClassifierService(enabled=True, model_path=model_path)
            ml_features = build_ml_features(
                question="Calculate the simple interest on ₹3000 at 4% for 2 years.",
                chapter="Simple Interest",
            )
            prediction = ml_service.predict(ml_features)

            self.assertIsNotNone(prediction)
            self.assertEqual(list(prediction.keys()), CATEGORY_KEYS)

    def test_ml_blending_keeps_total_100(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            model_path = Path(temp_dir) / "model.json"
            write_synthetic_dataset(path=dataset_path)
            train_model_from_dataset(dataset_path=dataset_path, model_path=model_path)

            result = calculate_question_skill_weightage(
                question="Study the following table and chart to analyze the data.",
                chapter="Data Interpretation",
                use_ml=True,
                ml_service=MLClassifierService(enabled=True, model_path=model_path),
            )

            self.assertEqual(sum(result.values()), 100)
            self.assertEqual(list(result.keys()), CATEGORY_KEYS)

    def test_response_header_indicates_rule_plus_ml_mode(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            model_path = Path(temp_dir) / "model.json"
            write_synthetic_dataset(path=dataset_path)
            train_model_from_dataset(dataset_path=dataset_path, model_path=model_path)

            original_enabled = environ.get("QUESTION_ANALYSIS_ML_ENABLED")
            original_model_path = environ.get("QUESTION_ANALYSIS_ML_MODEL_PATH")

            environ["QUESTION_ANALYSIS_ML_ENABLED"] = "true"
            environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = str(model_path)

            try:
                response = self.client.post(
                    "/api/v1/question-analysis/analyze",
                    json={
                        "question": "Calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
                        "chapter": "Simple Interest",
                        "use_ml": True,
                    },
                )
            finally:
                if original_enabled is None:
                    environ.pop("QUESTION_ANALYSIS_ML_ENABLED", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_ENABLED"] = original_enabled

                if original_model_path is None:
                    environ.pop("QUESTION_ANALYSIS_ML_MODEL_PATH", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = original_model_path

            self.assertEqual(response.headers.get("X-Question-Analysis-Mode"), "rule_plus_ml")

    def test_request_body_use_ml_true_works_without_env_enable_flag(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            model_path = Path(temp_dir) / "model.json"
            write_synthetic_dataset(path=dataset_path)
            train_model_from_dataset(dataset_path=dataset_path, model_path=model_path)

            original_enabled = environ.get("QUESTION_ANALYSIS_ML_ENABLED")
            original_model_path = environ.get("QUESTION_ANALYSIS_ML_MODEL_PATH")

            environ.pop("QUESTION_ANALYSIS_ML_ENABLED", None)
            environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = str(model_path)

            try:
                response = self.client.post(
                    "/api/v1/question-analysis/analyze",
                    json={
                        "question": "Calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
                        "chapter": "Simple Interest",
                        "use_ml": True,
                    },
                )
            finally:
                if original_enabled is None:
                    environ.pop("QUESTION_ANALYSIS_ML_ENABLED", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_ENABLED"] = original_enabled

                if original_model_path is None:
                    environ.pop("QUESTION_ANALYSIS_ML_MODEL_PATH", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = original_model_path

            self.assertEqual(response.headers.get("X-Question-Analysis-Mode"), "rule_plus_ml")

    def test_request_body_switch_can_force_rule_only_even_when_ml_enabled(self):
        with TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            model_path = Path(temp_dir) / "model.json"
            write_synthetic_dataset(path=dataset_path)
            train_model_from_dataset(dataset_path=dataset_path, model_path=model_path)

            original_enabled = environ.get("QUESTION_ANALYSIS_ML_ENABLED")
            original_model_path = environ.get("QUESTION_ANALYSIS_ML_MODEL_PATH")

            environ["QUESTION_ANALYSIS_ML_ENABLED"] = "true"
            environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = str(model_path)

            try:
                response = self.client.post(
                    "/api/v1/question-analysis/analyze",
                    json={
                        "question": "Calculate the simple interest on ₹5000 at 5% per annum for 2 years.",
                        "chapter": "Simple Interest",
                        "use_ml": False,
                    },
                )
            finally:
                if original_enabled is None:
                    environ.pop("QUESTION_ANALYSIS_ML_ENABLED", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_ENABLED"] = original_enabled

                if original_model_path is None:
                    environ.pop("QUESTION_ANALYSIS_ML_MODEL_PATH", None)
                else:
                    environ["QUESTION_ANALYSIS_ML_MODEL_PATH"] = original_model_path

            self.assertEqual(response.headers.get("X-Question-Analysis-Mode"), "rule_only")


if __name__ == "__main__":
    unittest.main()
