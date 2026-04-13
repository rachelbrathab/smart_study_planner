from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ml_pipeline.utils.config import REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class MLTaskTypeStep:
    """Step 2: define the ML task type."""

    output_file: Path = REPORTS_DIR / "step_02_task_type.json"

    def run(self) -> Dict[str, str]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_02_task_type")

        result = {
            "step": "Step 2",
            "name": "ML Task Type",
            "task_type": "classification",
            "classification_type": "binary",
            "justification": "The target session_completed has two classes: 0 (not completed) and 1 (completed).",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        saved_path = save_json(result, self.output_file)
        logger.info("Step 2 completed. Task type artifact saved to %s", saved_path)
        return result


if __name__ == "__main__":
    step = MLTaskTypeStep()
    output = step.run()
    print("Step 2 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
