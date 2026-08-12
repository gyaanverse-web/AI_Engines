import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_04_context_and_step_itr2.question_analysis.synthetic_dataset import (
    write_synthetic_dataset,
)


if __name__ == "__main__":
    path = write_synthetic_dataset()
    print(path)
