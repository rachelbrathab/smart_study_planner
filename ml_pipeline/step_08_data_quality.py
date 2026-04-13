from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DataQualityStep:
    """Step 8: validate dataset quality (duplicates, dtypes, ranges)."""

    input_file: Path = DATA_DIR / "study_sessions_raw.csv"
    report_file: Path = REPORTS_DIR / "step_08_data_quality.json"

    def _expected_ranges(self) -> Dict[str, tuple]:
        return {
            "study_hours": (0.0, 8.0),
            "session_duration": (10.0, 180.0),
            "breaks_taken": (0.0, 12.0),
            "productivity_score": (0.0, 100.0),
            "session_completed": (0.0, 1.0),
        }

    def _range_violations(self, df: pd.DataFrame) -> Dict[str, int]:
        violations: Dict[str, int] = {}
        for col, (min_val, max_val) in self._expected_ranges().items():
            if col not in df.columns:
                violations[col] = -1
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            count = int(((valid < min_val) | (valid > max_val)).sum())
            violations[col] = count
        return violations

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_08_data_quality")

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input dataset not found at {self.input_file}. Run Step 6 first."
            )

        df = pd.read_csv(self.input_file)

        duplicate_rows = int(df.duplicated().sum())
        dtype_summary = {column: str(dtype) for column, dtype in df.dtypes.items()}
        missing_summary = {column: int(count) for column, count in df.isna().sum().to_dict().items()}
        range_violations = self._range_violations(df)

        checks: List[Dict[str, object]] = [
            {"name": "duplicate_rows", "passed": duplicate_rows == 0, "value": duplicate_rows},
            {
                "name": "session_completed_binary",
                "passed": set(pd.Series(df["session_completed"]).dropna().unique()).issubset({0, 1}),
                "value": sorted(pd.Series(df["session_completed"]).dropna().unique().tolist()),
            },
            {
                "name": "range_violations",
                "passed": all(v == 0 for v in range_violations.values() if v >= 0),
                "value": range_violations,
            },
        ]

        quality_passed = all(item["passed"] for item in checks)

        report = {
            "step": "Step 8",
            "name": "Data Quality",
            "input_dataset": str(self.input_file),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicate_rows": duplicate_rows,
            "dtype_summary": dtype_summary,
            "missing_values_by_column": missing_summary,
            "range_violations_by_column": range_violations,
            "checks": checks,
            "quality_passed": quality_passed,
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 8 completed. Data quality report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = DataQualityStep()
    output = step.run()
    print("Step 8 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
