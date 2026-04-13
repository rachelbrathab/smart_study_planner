from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ml_pipeline.utils.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class HyperparameterTuningStep:
    """Step 18: tune Random Forest hyperparameters with GridSearchCV."""

    x_train_file: Path = DATA_DIR / "train_transformed.csv"
    y_train_file: Path = DATA_DIR / "y_train.csv"
    x_val_file: Path = DATA_DIR / "validation_transformed.csv"
    y_val_file: Path = DATA_DIR / "y_validation.csv"

    tuned_model_output: Path = MODELS_DIR / "tuned_random_forest.joblib"
    preds_output: Path = REPORTS_DIR / "step_18_validation_predictions.csv"
    report_output: Path = REPORTS_DIR / "step_18_tuning.json"

    target_column: str = "session_completed"

    def _load_xy(self):
        required = [self.x_train_file, self.y_train_file, self.x_val_file, self.y_val_file]
        for fp in required:
            if not fp.exists():
                raise FileNotFoundError(f"Missing required input file: {fp}")

        x_train = pd.read_csv(self.x_train_file)
        y_train = pd.read_csv(self.y_train_file)[self.target_column]
        x_val = pd.read_csv(self.x_val_file)
        y_val = pd.read_csv(self.y_val_file)[self.target_column]
        return x_train, y_train, x_val, y_val

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_18_tuning")

        x_train, y_train, x_val, y_val = self._load_xy()

        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [150, 300],
            "max_depth": [None, 8, 12],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1",
            n_jobs=-1,
            cv=cv,
            verbose=0,
        )
        grid.fit(x_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_val)

        metrics = {
            "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
            "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        }

        preds_df = pd.DataFrame({"y_true": y_val.reset_index(drop=True), "y_pred": y_pred})
        preds_df.to_csv(self.preds_output, index=False)
        joblib.dump(best_model, self.tuned_model_output)

        report = {
            "step": "Step 18",
            "name": "Hyperparameter Tuning",
            "search_method": "GridSearchCV",
            "cv_folds": 3,
            "scoring": "f1",
            "param_grid": param_grid,
            "best_params": grid.best_params_,
            "best_cv_score_f1": round(float(grid.best_score_), 4),
            "validation_metrics": metrics,
            "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "validation_shape": [int(x_val.shape[0]), int(x_val.shape[1])],
            "output_files": {
                "tuned_model": str(self.tuned_model_output),
                "validation_predictions": str(self.preds_output),
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_output)
        logger.info("Step 18 completed. Tuned model saved to %s", self.tuned_model_output)
        logger.info("Step 18 completed. Tuning report saved to %s", self.report_output)
        return report


if __name__ == "__main__":
    step = HyperparameterTuningStep()
    output = step.run()
    print("Step 18 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
