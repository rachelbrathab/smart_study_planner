from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml_pipeline.utils.config import DATA_DIR, PLOTS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class EDAStep:
    """Step 7: run exploratory data analysis and save summary reports/plots."""

    input_file: Path = DATA_DIR / "study_sessions_raw.csv"
    report_file: Path = REPORTS_DIR / "step_07_eda.json"
    summary_csv: Path = REPORTS_DIR / "step_07_eda_numeric_summary.csv"

    def _save_class_distribution_plot(self, df: pd.DataFrame) -> Path:
        output_path = PLOTS_DIR / "eda_class_distribution.png"
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="session_completed", hue="session_completed", palette="Set2", legend=False)
        plt.title("Session Completed Class Distribution")
        plt.xlabel("session_completed")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path

    def _save_numeric_distributions_plot(self, df: pd.DataFrame) -> Path:
        output_path = PLOTS_DIR / "eda_numeric_distributions.png"
        numeric_cols = ["study_hours", "session_duration", "breaks_taken", "productivity_score"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.flatten()
        for idx, col in enumerate(numeric_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[idx], color="#2E7D32")
            axes[idx].set_title(f"Distribution: {col}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def _save_correlation_plot(self, df: pd.DataFrame) -> Path:
        output_path = PLOTS_DIR / "eda_correlation_heatmap.png"
        corr_cols = ["study_hours", "session_duration", "breaks_taken", "productivity_score", "session_completed"]
        corr = df[corr_cols].corr(numeric_only=True)

        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_07_eda")

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input dataset not found at {self.input_file}. Run Step 6 first."
            )

        df = pd.read_csv(self.input_file)

        numeric_summary = df.describe(include="number").round(3)
        numeric_summary.to_csv(self.summary_csv, index=True)

        class_plot = self._save_class_distribution_plot(df)
        numeric_plot = self._save_numeric_distributions_plot(df)
        corr_plot = self._save_correlation_plot(df)

        missing_by_column = {col: int(val) for col, val in df.isna().sum().to_dict().items()}

        report = {
            "step": "Step 7",
            "name": "Exploratory Data Analysis",
            "input_dataset": str(self.input_file),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_summary_csv": str(self.summary_csv),
            "missing_values_by_column": missing_by_column,
            "plots_generated": [
                str(class_plot),
                str(numeric_plot),
                str(corr_plot),
            ],
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 7 completed. EDA report saved to %s", self.report_file)
        logger.info("Step 7 completed. EDA plots saved to %s", PLOTS_DIR)
        return report


if __name__ == "__main__":
    step = EDAStep()
    output = step.run()
    print("Step 7 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
