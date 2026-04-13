from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from ml_pipeline.utils.config import MODELS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DeploymentStep:
    """Step 20: package artifacts and simulate a prediction API call."""

    model_file: Path = MODELS_DIR / "tuned_random_forest.joblib"
    preprocessor_file: Path = MODELS_DIR / "preprocessor_step_15.joblib"
    package_file: Path = MODELS_DIR / "deployment_bundle.joblib"
    sample_output_file: Path = REPORTS_DIR / "step_20_sample_predictions.csv"
    report_file: Path = REPORTS_DIR / "step_20_deployment.json"

    def _engineer_features(self, payload_df: pd.DataFrame) -> pd.DataFrame:
        out = payload_df.copy()
        out["efficiency"] = out["productivity_score"] / out["study_hours"].replace(0, 1e-6)
        out["study_intensity"] = out["session_duration"] / (out["breaks_taken"] + 1)
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
        out["difficulty_code"] = out["subject_difficulty"].map(difficulty_map).fillna(2)
        out["difficulty_burden"] = out["session_duration"] * out["difficulty_code"]
        out["productive_pace"] = out["productivity_score"] / out["session_duration"].replace(0, 1e-6)
        return out

    def _simulate_prediction_api(self, model, preprocessor, payloads: List[Dict[str, object]]) -> pd.DataFrame:
        payload_df = pd.DataFrame(payloads)
        engineered = self._engineer_features(payload_df)
        x_tx = preprocessor.transform(engineered)
        feature_names = list(getattr(model, "feature_names_in_", []))
        if feature_names:
            x_tx = pd.DataFrame(x_tx, columns=feature_names)

        preds = model.predict(x_tx)
        probs = model.predict_proba(x_tx)[:, 1]

        result = payload_df.copy()
        result["predicted_session_completed"] = preds
        result["prob_session_completed"] = probs.round(4)
        return result

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_20_deployment")

        for fp in [self.model_file, self.preprocessor_file]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing required artifact: {fp}")

        model = joblib.load(self.model_file)
        preprocessor = joblib.load(self.preprocessor_file)

        deployment_bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "model_name": "tuned_random_forest",
            "created_at": utc_timestamp(),
            "api_contract": {
                "required_features": [
                    "study_hours",
                    "session_duration",
                    "breaks_taken",
                    "subject_difficulty",
                    "time_of_day",
                    "productivity_score",
                ],
                "prediction_field": "predicted_session_completed",
            },
        }
        joblib.dump(deployment_bundle, self.package_file)

        sample_payloads = [
            {
                "study_hours": 3.8,
                "session_duration": 80,
                "breaks_taken": 2,
                "subject_difficulty": "medium",
                "time_of_day": "morning",
                "productivity_score": 86,
            },
            {
                "study_hours": 1.4,
                "session_duration": 40,
                "breaks_taken": 5,
                "subject_difficulty": "hard",
                "time_of_day": "night",
                "productivity_score": 48,
            },
        ]

        sample_predictions = self._simulate_prediction_api(model, preprocessor, sample_payloads)
        sample_predictions.to_csv(self.sample_output_file, index=False)

        report = {
            "step": "Step 20",
            "name": "Deployment",
            "deployment_type": "simulation",
            "artifacts_used": {
                "model": str(self.model_file),
                "preprocessor": str(self.preprocessor_file),
            },
            "deployment_bundle": str(self.package_file),
            "sample_prediction_output": str(self.sample_output_file),
            "api_simulation_status": "success",
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 20 completed. Deployment bundle saved to %s", self.package_file)
        logger.info("Step 20 completed. Deployment report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = DeploymentStep()
    output = step.run()
    print("Step 20 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
