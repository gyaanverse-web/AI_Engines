import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from question_analysis.ml_training import (
    train_model_from_dataset,
)
from question_analysis.synthetic_dataset import (
    write_synthetic_dataset,
)


if __name__ == "__main__":
    dataset_path = write_synthetic_dataset()
    model_path = train_model_from_dataset(dataset_path=dataset_path)
    print(model_path)
