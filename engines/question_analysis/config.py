import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DEFAULT_DATASET_PATH = DATA_DIR / "synthetic_question_skill_dataset.jsonl"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "question_skill_model.json"

ML_ENABLED_ENV_VAR = "QUESTION_ANALYSIS_ML_ENABLED"
ML_MODEL_PATH_ENV_VAR = "QUESTION_ANALYSIS_ML_MODEL_PATH"
ML_RULE_WEIGHT_ENV_VAR = "QUESTION_ANALYSIS_RULE_WEIGHT"
ML_MODEL_WEIGHT_ENV_VAR = "QUESTION_ANALYSIS_MODEL_WEIGHT"


def is_ml_enabled() -> bool:
    return os.getenv(ML_ENABLED_ENV_VAR, "false").strip().lower() in {"1", "true", "yes", "on"}


def get_model_path() -> Path:
    configured_path = os.getenv(ML_MODEL_PATH_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_PATH


def get_rule_weight() -> float:
    return float(os.getenv(ML_RULE_WEIGHT_ENV_VAR, "0.6"))


def get_model_weight() -> float:
    return float(os.getenv(ML_MODEL_WEIGHT_ENV_VAR, "0.4"))
