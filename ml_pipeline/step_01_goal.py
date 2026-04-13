from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ml_pipeline.utils.config import REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class BusinessGoalStep:
    """Step 1: define the business goal for the ML system."""

    output_file: Path = REPORTS_DIR / "step_01_goal.json"

    def run(self) -> Dict[str, str]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_01_goal")

        result = {
            "step": "Step 1",
            "name": "Business Goal",
            "problem_statement": "Predict whether a study session will be completed.",
            "business_value": "Helps students identify sessions at risk and improve planning outcomes.",
            "target_outcome": "Increase study-session completion consistency over time.",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        saved_path = save_json(result, self.output_file)
        logger.info("Step 1 completed. Goal artifact saved to %s", saved_path)
        return result


if __name__ == "__main__":
    step = BusinessGoalStep()
    output = step.run()
    print("Step 1 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
