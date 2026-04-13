from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ml_pipeline.utils.config import REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class TargetLabelStep:
    """Step 3: define the target label for model training."""

    output_file: Path = REPORTS_DIR / "step_03_target.json"

    def run(self) -> Dict[str, str]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_03_target")

        result = {
            "step": "Step 3",
            "name": "Target Label",
            "target_column": "session_completed",
            "target_type": "binary",
            "negative_class": "0 = session not completed",
            "positive_class": "1 = session completed",
            "justification": "This target directly represents study session outcome and supports binary classification.",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        saved_path = save_json(result, self.output_file)
        logger.info("Step 3 completed. Target artifact saved to %s", saved_path)
        return result


if __name__ == "__main__":
    step = TargetLabelStep()
    output = step.run()
    print("Step 3 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
