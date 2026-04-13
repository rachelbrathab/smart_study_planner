from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.utils.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class TransformationStep:
    """Step 15: apply scaling + encoding and save transformed datasets."""

    train_input: Path = DATA_DIR / "train_engineered.csv"
    val_input: Path = DATA_DIR / "validation_engineered.csv"
    test_input: Path = DATA_DIR / "test_engineered.csv"

    train_output: Path = DATA_DIR / "train_transformed.csv"
    val_output: Path = DATA_DIR / "validation_transformed.csv"
    test_output: Path = DATA_DIR / "test_transformed.csv"

    y_train_output: Path = DATA_DIR / "y_train.csv"
    y_val_output: Path = DATA_DIR / "y_validation.csv"
    y_test_output: Path = DATA_DIR / "y_test.csv"

    preprocessor_output: Path = MODELS_DIR / "preprocessor_step_15.joblib"
    report_file: Path = REPORTS_DIR / "step_15_transformation.json"

    target_column: str = "session_completed"

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        return pd.read_csv(path)

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found.")
        x_df = df.drop(columns=[self.target_column])
        y = pd.to_numeric(df[self.target_column], errors="coerce").astype(int)
        return x_df, y

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_15_transformation")

        train_df = self._load(self.train_input)
        val_df = self._load(self.val_input)
        test_df = self._load(self.test_input)

        x_train, y_train = self._split_xy(train_df)
        x_val, y_val = self._split_xy(val_df)
        x_test, y_test = self._split_xy(test_df)

        numeric_features: List[str] = [
            "study_hours",
            "session_duration",
            "breaks_taken",
            "productivity_score",
            "efficiency",
            "study_intensity",
            "difficulty_code",
            "difficulty_burden",
            "productive_pace",
        ]
        categorical_features: List[str] = ["subject_difficulty", "time_of_day"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("scaler", StandardScaler())]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        x_train_tx = preprocessor.fit_transform(x_train)
        x_val_tx = preprocessor.transform(x_val)
        x_test_tx = preprocessor.transform(x_test)

        feature_names = preprocessor.get_feature_names_out().tolist()

        train_tx_df = pd.DataFrame(x_train_tx, columns=feature_names)
        val_tx_df = pd.DataFrame(x_val_tx, columns=feature_names)
        test_tx_df = pd.DataFrame(x_test_tx, columns=feature_names)

        train_tx_df.to_csv(self.train_output, index=False)
        val_tx_df.to_csv(self.val_output, index=False)
        test_tx_df.to_csv(self.test_output, index=False)

        y_train.to_frame(self.target_column).to_csv(self.y_train_output, index=False)
        y_val.to_frame(self.target_column).to_csv(self.y_val_output, index=False)
        y_test.to_frame(self.target_column).to_csv(self.y_test_output, index=False)

        joblib.dump(preprocessor, self.preprocessor_output)

        report = {
            "step": "Step 15",
            "name": "Transformation",
            "inputs": {
                "train": str(self.train_input),
                "validation": str(self.val_input),
                "test": str(self.test_input),
            },
            "outputs": {
                "x_train": str(self.train_output),
                "x_validation": str(self.val_output),
                "x_test": str(self.test_output),
                "y_train": str(self.y_train_output),
                "y_validation": str(self.y_val_output),
                "y_test": str(self.y_test_output),
                "preprocessor": str(self.preprocessor_output),
            },
            "numeric_features_scaled": numeric_features,
            "categorical_features_encoded": categorical_features,
            "transformed_feature_count": len(feature_names),
            "transformed_feature_names": feature_names,
            "shapes": {
                "x_train": [int(train_tx_df.shape[0]), int(train_tx_df.shape[1])],
                "x_validation": [int(val_tx_df.shape[0]), int(val_tx_df.shape[1])],
                "x_test": [int(test_tx_df.shape[0]), int(test_tx_df.shape[1])],
            },
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_file)
        logger.info("Step 15 completed. Transformed datasets saved to %s", DATA_DIR)
        logger.info("Step 15 completed. Preprocessor saved to %s", self.preprocessor_output)
        logger.info("Step 15 completed. Transformation report saved to %s", self.report_file)
        return report


if __name__ == "__main__":
    step = TransformationStep()
    output = step.run()
    print("Step 15 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
