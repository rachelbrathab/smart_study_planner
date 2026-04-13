from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ml_pipeline.utils.config import REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DataSourceStep:
    """Step 5: define the data source strategy for the ML pipeline."""

    output_file: Path = REPORTS_DIR / "step_05_data_source.json"

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_05_data_source")

        feature_columns: List[str] = [
            "study_hours",
            "session_duration",
            "breaks_taken",
            "subject_difficulty",
            "time_of_day",
            "productivity_score",
        ]

        result = {
            "step": "Step 5",
            "name": "Data Source",
            "source_type": "synthetic",
            "external_dataset_used": False,
            "generator_library_stack": ["numpy", "pandas", "scikit-learn"],
            "feature_columns": feature_columns,
            "target_column": "session_completed",
            "data_requirements": {
                "balanced_target": True,
                "contains_noise": True,
                "contains_missing_values": True,
            },
            "justification": "Synthetic data is required by project constraints and enables controlled testing of the complete ML pipeline.",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        saved_path = save_json(result, self.output_file)
        logger.info("Step 5 completed. Data source artifact saved to %s", saved_path)
        return result


if __name__ == "__main__":
    step = DataSourceStep()
    output = step.run()
    print("Step 5 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
