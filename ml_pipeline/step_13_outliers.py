from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class OutlierStep:
    """Step 13: detect and cap outliers using IQR-based bounds."""

    train_input: Path = DATA_DIR / "train_missing_handled.csv"
    val_input: Path = DATA_DIR / "validation_missing_handled.csv"
    test_input: Path = DATA_DIR / "test_missing_handled.csv"

    train_output: Path = DATA_DIR / "train_outliers_handled.csv"
    val_output: Path = DATA_DIR / "validation_outliers_handled.csv"
    test_output: Path = DATA_DIR / "test_outliers_handled.csv"

    report_file: Path = REPORTS_DIR / "step_13_outliers.json"

    def _load(self, file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")
        return pd.read_csv(file_path)

    def _iqr_bounds(self, train_df: pd.DataFrame, columns: list[str]) -> Dict[str, Dict[str, float]]:
        bounds: Dict[str, Dict[str, float]] = {}
        for col in columns:
            q1 = float(train_df[col].quantile(0.25))
            q3 = float(train_df[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            bounds[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
            }
        return bounds

    def _count_outliers(self, df: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for col, b in bounds.items():
            count = int(((df[col] < b["lower"]) | (df[col] > b["upper"])).sum())
            result[col] = count
        return result

    def _cap_outliers(self, df: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        out = df.copy()
        for col, b in bounds.items():
            out[col] = out[col].clip(lower=b["lower"], upper=b["upper"])
        return out

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_13_outliers")

        train_df = self._load(self.train_input)
        val_df = self._load(self.val_input)
        test_df = self._load(self.test_input)

        numeric_cols = ["study_hours", "session_duration", "breaks_taken", "productivity_score"]
        bounds = self._iqr_bounds(train_df, numeric_cols)

        before_counts = {
            "train": self._count_outliers(train_df, bounds),
            "validation": self._count_outliers(val_df, bounds),
            "test": self._count_outliers(test_df, bounds),
        }

        train_out = self._cap_outliers(train_df, bounds)
        val_out = self._cap_outliers(val_df, bounds)
        test_out = self._cap_outliers(test_df, bounds)

        after_counts = {
            "train": self._count_outliers(train_out, bounds),
            "validation": self._count_outliers(val_out, bounds),
            "test": self._count_outliers(test_out, bounds),
        }

        train_out.to_csv(self.train_output, index=False)
        val_out.to_csv(self.val_output, index=False)
        test_out.to_csv(self.test_output, index=False)

        report = {
            "step": "Step 13",
            "name": "Outliers",
            "method": "iqr_capping",
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
            "columns_checked": numeric_cols,
            "train_fitted_bounds": bounds,
            "outlier_counts_before": before_counts,
            "outlier_counts_after": after_counts,
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 13 completed. Outlier-handled files saved to %s", DATA_DIR)
        logger.info("Step 13 completed. Outlier report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = OutlierStep()
    output = step.run()
    print("Step 13 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
