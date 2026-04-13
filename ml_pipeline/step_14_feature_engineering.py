from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class FeatureEngineeringStep:
    """Step 14: create additional predictive features."""

    train_input: Path = DATA_DIR / "train_outliers_handled.csv"
    val_input: Path = DATA_DIR / "validation_outliers_handled.csv"
    test_input: Path = DATA_DIR / "test_outliers_handled.csv"

    train_output: Path = DATA_DIR / "train_engineered.csv"
    val_output: Path = DATA_DIR / "validation_engineered.csv"
    test_output: Path = DATA_DIR / "test_engineered.csv"

    report_file: Path = REPORTS_DIR / "step_14_feature_engineering.json"

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        return pd.read_csv(path)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Efficiency: productivity score per study hour.
        out["efficiency"] = out["productivity_score"] / out["study_hours"].replace(0, 1e-6)

        # Study intensity: effective study minutes per break.
        out["study_intensity"] = out["session_duration"] / (out["breaks_taken"] + 1)

        # Difficulty burden: combine time spent and subject hardness.
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
        out["difficulty_code"] = out["subject_difficulty"].map(difficulty_map).fillna(2)
        out["difficulty_burden"] = out["session_duration"] * out["difficulty_code"]

        # Productive pace: score gained per minute.
        out["productive_pace"] = out["productivity_score"] / out["session_duration"].replace(0, 1e-6)

        return out

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_14_feature_engineering")

        train_df = self._load(self.train_input)
        val_df = self._load(self.val_input)
        test_df = self._load(self.test_input)

        train_out = self._engineer(train_df)
        val_out = self._engineer(val_df)
        test_out = self._engineer(test_df)

        train_out.to_csv(self.train_output, index=False)
        val_out.to_csv(self.val_output, index=False)
        test_out.to_csv(self.test_output, index=False)

        new_features: List[str] = [
            "efficiency",
            "study_intensity",
            "difficulty_code",
            "difficulty_burden",
            "productive_pace",
        ]

        report = {
            "step": "Step 14",
            "name": "Feature Engineering",
            "inputs": {
                "train": str(self.train_input),
                "validation": str(self.val_input),
                "test": str(self.test_input),
            },
            "outputs": {
                "train": str(self.train_output),
                "validation": str(self.val_output),
                "test": str(self.test_output),
            },
            "new_features_created": new_features,
            "shape_after_engineering": {
                "train": [int(train_out.shape[0]), int(train_out.shape[1])],
                "validation": [int(val_out.shape[0]), int(val_out.shape[1])],
                "test": [int(test_out.shape[0]), int(test_out.shape[1])],
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 14 completed. Engineered datasets saved to %s", DATA_DIR)
        logger.info("Step 14 completed. Feature engineering report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = FeatureEngineeringStep()
    output = step.run()
    print("Step 14 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
