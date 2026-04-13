from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml_pipeline.utils.config import DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class EvaluationStep:
    """Step 19: evaluate tuned model on holdout test set."""

    x_test_file: Path = DATA_DIR / "test_transformed.csv"
    y_test_file: Path = DATA_DIR / "y_test.csv"
    model_file: Path = MODELS_DIR / "tuned_random_forest.joblib"

    confusion_plot_file: Path = PLOTS_DIR / "evaluation_confusion_matrix.png"
    feature_importance_file: Path = REPORTS_DIR / "step_19_feature_importance.csv"
    predictions_file: Path = REPORTS_DIR / "step_19_test_predictions.csv"
    report_file: Path = REPORTS_DIR / "step_19_evaluation.json"

    target_column: str = "session_completed"

    def _load_inputs(self):
        for fp in [self.x_test_file, self.y_test_file, self.model_file]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing required input file: {fp}")

        x_test = pd.read_csv(self.x_test_file)
        y_test = pd.read_csv(self.y_test_file)[self.target_column]
        model = joblib.load(self.model_file)
        return x_test, y_test, model

    def _save_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, int]:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(self.confusion_plot_file, dpi=160)
        plt.close()

        return {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_19_evaluation")

        x_test, y_test, model = self._load_inputs()
        y_pred = model.predict(x_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }

        confusion = self._save_confusion_matrix(y_test, y_pred)

        preds_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_pred": y_pred})
        preds_df.to_csv(self.predictions_file, index=False)

        feature_importance_df = pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(self.feature_importance_file, index=False)

        report = {
            "step": "Step 19",
            "name": "Evaluation",
            "model_evaluated": str(self.model_file),
            "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
            "metrics_test": metrics,
            "confusion_matrix": confusion,
            "output_files": {
                "confusion_matrix_plot": str(self.confusion_plot_file),
                "feature_importance": str(self.feature_importance_file),
                "test_predictions": str(self.predictions_file),
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 19 completed. Evaluation report saved to %s", self.report_file)
        logger.info("Step 19 completed. Confusion matrix plot saved to %s", self.confusion_plot_file)
        return report


if __name__ == "__main__":
    step = EvaluationStep()
    output = step.run()
    print("Step 19 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
