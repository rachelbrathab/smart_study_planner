from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml_pipeline.utils.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class BaselineModelStep:
    """Step 16: train and evaluate baseline Logistic Regression model."""

    x_train_file: Path = DATA_DIR / "train_transformed.csv"
    y_train_file: Path = DATA_DIR / "y_train.csv"
    x_val_file: Path = DATA_DIR / "validation_transformed.csv"
    y_val_file: Path = DATA_DIR / "y_validation.csv"

    model_output: Path = MODELS_DIR / "baseline_logistic_regression.joblib"
    preds_output: Path = REPORTS_DIR / "step_16_validation_predictions.csv"
    report_output: Path = REPORTS_DIR / "step_16_baseline.json"

    target_column: str = "session_completed"

    def _load_xy(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        for fp in [self.x_train_file, self.y_train_file, self.x_val_file, self.y_val_file]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing required input file: {fp}")

        x_train = pd.read_csv(self.x_train_file)
        y_train = pd.read_csv(self.y_train_file)[self.target_column]
        x_val = pd.read_csv(self.x_val_file)
        y_val = pd.read_csv(self.y_val_file)[self.target_column]

        return x_train, y_train, x_val, y_val

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_16_baseline")

        x_train, y_train, x_val, y_val = self._load_xy()

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)

        metrics = {
            "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
            "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        }

        preds_df = pd.DataFrame(
            {
                "y_true": y_val.reset_index(drop=True),
                "y_pred": pd.Series(y_pred).reset_index(drop=True),
            }
        )
        preds_df.to_csv(self.preds_output, index=False)
        joblib.dump(model, self.model_output)

        report = {
            "step": "Step 16",
            "name": "Baseline Model",
            "model_type": "LogisticRegression",
            "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "validation_shape": [int(x_val.shape[0]), int(x_val.shape[1])],
            "metrics_validation": metrics,
            "output_files": {
                "model": str(self.model_output),
                "validation_predictions": str(self.preds_output),
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_output)
        logger.info("Step 16 completed. Baseline model saved to %s", self.model_output)
        logger.info("Step 16 completed. Baseline report saved to %s", self.report_output)
        return report


if __name__ == "__main__":
    step = BaselineModelStep()
    output = step.run()
    print("Step 16 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
