from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class MissingValuesStep:
    """Step 12: handle missing values using train-fitted imputation."""

    train_input: Path = DATA_DIR / "train_clean.csv"
    val_input: Path = DATA_DIR / "validation_clean.csv"
    test_input: Path = DATA_DIR / "test_clean.csv"

    train_output: Path = DATA_DIR / "train_missing_handled.csv"
    val_output: Path = DATA_DIR / "validation_missing_handled.csv"
    test_output: Path = DATA_DIR / "test_missing_handled.csv"

    report_file: Path = REPORTS_DIR / "step_12_missing.json"

    def _load(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")
        return pd.read_csv(file_path)

    def _missing_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        return {k: int(v) for k, v in df.isna().sum().to_dict().items()}

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_12_missing")

        train_df = self._load(self.train_input)
        val_df = self._load(self.val_input)
        test_df = self._load(self.test_input)

        before_missing = {
            "train": self._missing_counts(train_df),
            "validation": self._missing_counts(val_df),
            "test": self._missing_counts(test_df),
        }

        numeric_columns = ["study_hours", "session_duration", "breaks_taken", "productivity_score"]
        categorical_columns = ["subject_difficulty", "time_of_day"]

        # Fit imputation statistics on train only to avoid leakage.
        numeric_impute_values = {
            col: float(train_df[col].median()) for col in numeric_columns if col in train_df.columns
        }

        categorical_impute_values = {}
        for col in categorical_columns:
            if col in train_df.columns:
                mode = train_df[col].mode(dropna=True)
                categorical_impute_values[col] = str(mode.iloc[0]) if not mode.empty else "unknown"

        def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col, val in numeric_impute_values.items():
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(val)
            for col, val in categorical_impute_values.items():
                if col in out.columns:
                    out[col] = out[col].fillna(val)
            return out

        train_out = apply_imputation(train_df)
        val_out = apply_imputation(val_df)
        test_out = apply_imputation(test_df)

        train_out.to_csv(self.train_output, index=False)
        val_out.to_csv(self.val_output, index=False)
        test_out.to_csv(self.test_output, index=False)

        after_missing = {
            "train": self._missing_counts(train_out),
            "validation": self._missing_counts(val_out),
            "test": self._missing_counts(test_out),
        }

        report = {
            "step": "Step 12",
            "name": "Missing Values",
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
            "imputation_strategy": {
                "numeric": "median",
                "categorical": "mode",
            },
            "numeric_impute_values": numeric_impute_values,
            "categorical_impute_values": categorical_impute_values,
            "missing_before": before_missing,
            "missing_after": after_missing,
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 12 completed. Missing-value handled files saved to %s", DATA_DIR)
        logger.info("Step 12 completed. Missing-value report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = MissingValuesStep()
    output = step.run()
    print("Step 12 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
