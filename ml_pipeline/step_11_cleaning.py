from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DataCleaningStep:
    """Step 11: clean split datasets by removing duplicates and fixing dtypes."""

    train_input: Path = DATA_DIR / "train_raw.csv"
    val_input: Path = DATA_DIR / "validation_raw.csv"
    test_input: Path = DATA_DIR / "test_raw.csv"

    train_output: Path = DATA_DIR / "train_clean.csv"
    val_output: Path = DATA_DIR / "validation_clean.csv"
    test_output: Path = DATA_DIR / "test_clean.csv"

    report_file: Path = REPORTS_DIR / "step_11_cleaning.json"

    def _clean_frame(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        before_rows = int(df.shape[0])
        duplicate_rows = int(df.duplicated().sum())

        clean_df = df.drop_duplicates().copy()

        numeric_columns = [
            "study_hours",
            "session_duration",
            "breaks_taken",
            "productivity_score",
            "session_completed",
        ]
        for col in numeric_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        if "session_completed" in clean_df.columns:
            clean_df["session_completed"] = clean_df["session_completed"].round().astype("Int64")

        categorical_columns = ["subject_difficulty", "time_of_day"]
        for col in categorical_columns:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str).str.strip().str.lower().replace({"nan": pd.NA})

        after_rows = int(clean_df.shape[0])
        dtype_summary = {k: str(v) for k, v in clean_df.dtypes.items()}

        summary = {
            "rows_before": before_rows,
            "rows_after": after_rows,
            "duplicates_removed": duplicate_rows,
            "dtype_summary": dtype_summary,
        }
        return clean_df, summary

    def _load_required(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")
        return pd.read_csv(file_path)

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_11_cleaning")

        train_df = self._load_required(self.train_input)
        val_df = self._load_required(self.val_input)
        test_df = self._load_required(self.test_input)

        train_clean, train_summary = self._clean_frame(train_df)
        val_clean, val_summary = self._clean_frame(val_df)
        test_clean, test_summary = self._clean_frame(test_df)

        train_clean.to_csv(self.train_output, index=False)
        val_clean.to_csv(self.val_output, index=False)
        test_clean.to_csv(self.test_output, index=False)

        report = {
            "step": "Step 11",
            "name": "Cleaning",
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
            "summary": {
                "train": train_summary,
                "validation": val_summary,
                "test": test_summary,
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 11 completed. Cleaned files saved to %s", DATA_DIR)
        logger.info("Step 11 completed. Cleaning report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = DataCleaningStep()
    output = step.run()
    print("Step 11 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
