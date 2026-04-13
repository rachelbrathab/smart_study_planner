from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class LabelQualityStep:
    """Step 9: validate target label quality and class balance."""

    input_file: Path = DATA_DIR / "study_sessions_raw.csv"
    report_file: Path = REPORTS_DIR / "step_09_label_quality.json"
    target_column: str = "session_completed"

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_09_label_quality")

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input dataset not found at {self.input_file}. Run Step 6 first."
            )

        df = pd.read_csv(self.input_file)
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")

        target_series = df[self.target_column]
        missing_target = int(target_series.isna().sum())

        unique_values = sorted(pd.Series(target_series.dropna().unique()).tolist())
        is_binary = set(unique_values).issubset({0, 1})

        class_counts_raw = target_series.value_counts(dropna=False).to_dict()
        class_counts = {str(k): int(v) for k, v in class_counts_raw.items()}

        non_null_counts = target_series.dropna().value_counts()
        min_count = int(non_null_counts.min()) if not non_null_counts.empty else 0
        max_count = int(non_null_counts.max()) if not non_null_counts.empty else 0
        imbalance_ratio = 0.0 if max_count == 0 else round(min_count / max_count, 4)

        # Balance is considered acceptable when minority class is at least 40% of majority class.
        class_balance_passed = imbalance_ratio >= 0.4

        checks = [
            {
                "name": "target_missing_values",
                "passed": missing_target == 0,
                "value": missing_target,
            },
            {
                "name": "target_binary_values",
                "passed": is_binary,
                "value": unique_values,
            },
            {
                "name": "target_class_balance",
                "passed": class_balance_passed,
                "value": {
                    "imbalance_ratio": imbalance_ratio,
                    "threshold": 0.4,
                },
            },
        ]

        label_quality_passed = all(item["passed"] for item in checks)

        report = {
            "step": "Step 9",
            "name": "Label Quality",
            "input_dataset": str(self.input_file),
            "target_column": self.target_column,
            "rows": int(df.shape[0]),
            "class_counts": class_counts,
            "unique_target_values": unique_values,
            "target_missing_values": missing_target,
            "imbalance_ratio": imbalance_ratio,
            "checks": checks,
            "label_quality_passed": label_quality_passed,
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 9 completed. Label quality report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = LabelQualityStep()
    output = step.run()
    print("Step 9 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
