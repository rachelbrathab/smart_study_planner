from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DataSplitStep:
    """Step 10: split data into train, validation, and test sets."""

    input_file: Path = DATA_DIR / "study_sessions_raw.csv"
    train_file: Path = DATA_DIR / "train_raw.csv"
    val_file: Path = DATA_DIR / "validation_raw.csv"
    test_file: Path = DATA_DIR / "test_raw.csv"
    report_file: Path = REPORTS_DIR / "step_10_split.json"
    target_column: str = "session_completed"
    random_state: int = 42

    def _class_dist(self, df: pd.DataFrame) -> Dict[str, int]:
        return {str(k): int(v) for k, v in df[self.target_column].value_counts().sort_index().to_dict().items()}

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_10_split")

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input dataset not found at {self.input_file}. Run Step 6 first."
            )

        df = pd.read_csv(self.input_file)
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found.")

        # 70% train, 15% validation, 15% test using two-stage stratified split.
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=self.random_state,
            stratify=temp_df[self.target_column],
        )

        train_df.to_csv(self.train_file, index=False)
        val_df.to_csv(self.val_file, index=False)
        test_df.to_csv(self.test_file, index=False)

        report = {
            "step": "Step 10",
            "name": "Data Split",
            "input_dataset": str(self.input_file),
            "split_strategy": "stratified_train_val_test",
            "target_column": self.target_column,
            "split_ratio": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "row_counts": {
                "train": int(train_df.shape[0]),
                "validation": int(val_df.shape[0]),
                "test": int(test_df.shape[0]),
            },
            "class_distribution": {
                "train": self._class_dist(train_df),
                "validation": self._class_dist(val_df),
                "test": self._class_dist(test_df),
            },
            "output_files": {
                "train": str(self.train_file),
                "validation": str(self.val_file),
                "test": str(self.test_file),
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 10 completed. Split files saved to %s", DATA_DIR)
        logger.info("Step 10 completed. Split report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = DataSplitStep()
    output = step.run()
    print("Step 10 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
