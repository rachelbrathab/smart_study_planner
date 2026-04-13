from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ml_pipeline.utils.config import REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class MetricsStep:
    """Step 4: define model evaluation metrics for binary classification."""

    output_file: Path = REPORTS_DIR / "step_04_metrics.json"

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_04_metrics")

        metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
        result = {
            "step": "Step 4",
            "name": "Evaluation Metrics",
            "task_type": "binary_classification",
            "metrics": metrics,
            "primary_metric": "f1",
            "justification": "F1 balances precision and recall, while accuracy, precision, and recall provide complementary performance views.",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        saved_path = save_json(result, self.output_file)
        logger.info("Step 4 completed. Metrics artifact saved to %s", saved_path)
        return result


if __name__ == "__main__":
    step = MetricsStep()
    output = step.run()
    print("Step 4 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
