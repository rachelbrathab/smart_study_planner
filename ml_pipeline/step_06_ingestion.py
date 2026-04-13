from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml_pipeline.utils.config import DATA_DIR, REPORTS_DIR, ensure_artifact_dirs
from ml_pipeline.utils.helpers import save_json, utc_timestamp
from ml_pipeline.utils.logger import get_logger


@dataclass
class DataIngestionStep:
    """Step 6: generate and save synthetic raw dataset."""

    n_samples: int = 1000
    random_state: int = 42
    data_output_file: Path = DATA_DIR / "study_sessions_raw.csv"
    report_output_file: Path = REPORTS_DIR / "step_06_ingestion.json"

    def _build_balanced_target(self) -> np.ndarray:
        half = self.n_samples // 2
        target = np.array([0] * half + [1] * (self.n_samples - half), dtype=int)
        return target

    def _inject_missing_values(self, df: pd.DataFrame, ratio: float = 0.05) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        missing_columns = ["study_hours", "breaks_taken", "productivity_score"]

        for column in missing_columns:
            missing_count = max(1, int(len(df) * ratio))
            missing_index = rng.choice(df.index.to_numpy(), size=missing_count, replace=False)
            df.loc[missing_index, column] = np.nan

        return df

    def _generate_synthetic_dataframe(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        target = self._build_balanced_target()

        study_hours = np.where(
            target == 1,
            rng.normal(loc=3.4, scale=0.7, size=self.n_samples),
            rng.normal(loc=2.2, scale=0.8, size=self.n_samples),
        )
        session_duration = np.where(
            target == 1,
            rng.normal(loc=72, scale=12, size=self.n_samples),
            rng.normal(loc=48, scale=14, size=self.n_samples),
        )
        breaks_taken = np.where(
            target == 1,
            rng.poisson(lam=2.2, size=self.n_samples),
            rng.poisson(lam=3.5, size=self.n_samples),
        )

        difficulty_map = np.array(["easy", "medium", "hard"])
        time_map = np.array(["morning", "afternoon", "evening", "night"])

        subject_difficulty = rng.choice(difficulty_map, size=self.n_samples, p=[0.25, 0.45, 0.30])
        time_of_day = rng.choice(time_map, size=self.n_samples, p=[0.30, 0.30, 0.30, 0.10])

        productivity_score = (
            35
            + (study_hours * 12)
            + (session_duration * 0.35)
            - (breaks_taken * 2.8)
            + (target * 8)
        )

        # Add controlled Gaussian noise so the dataset is realistic, not perfectly separable.
        study_hours = study_hours + rng.normal(0, 0.25, size=self.n_samples)
        session_duration = session_duration + rng.normal(0, 4.0, size=self.n_samples)
        productivity_score = productivity_score + rng.normal(0, 4.5, size=self.n_samples)

        df = pd.DataFrame(
            {
                "study_hours": np.clip(study_hours, 0.5, 6.5),
                "session_duration": np.clip(session_duration, 20, 120),
                "breaks_taken": np.clip(breaks_taken, 0, 8),
                "subject_difficulty": subject_difficulty,
                "time_of_day": time_of_day,
                "productivity_score": np.clip(productivity_score, 0, 100),
                "session_completed": target,
            }
        )

        df = shuffle(df, random_state=self.random_state).reset_index(drop=True)
        df = self._inject_missing_values(df, ratio=0.05)
        return df

    def run(self) -> Dict[str, object]:
        ensure_artifact_dirs()
        logger = get_logger("ml_pipeline.step_06_ingestion")

        df = self._generate_synthetic_dataframe()
        self.data_output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_output_file, index=False)

        class_counts = df["session_completed"].value_counts().sort_index().to_dict()
        missing_counts = df.isna().sum().to_dict()

        report = {
            "step": "Step 6",
            "name": "Data Ingestion",
            "source": "synthetic_generator",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "output_dataset_path": str(self.data_output_file),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "missing_values_by_column": {k: int(v) for k, v in missing_counts.items()},
            "created_at": utc_timestamp(),
            "status": "completed",
        }

        save_json(report, self.report_output_file)
        logger.info("Step 6 completed. Raw dataset saved to %s", self.data_output_file)
        logger.info("Step 6 completed. Ingestion report saved to %s", self.report_output_file)
        return report


if __name__ == "__main__":
    step = DataIngestionStep()
    output = step.run()
    print("Step 6 output:")
    for key, value in output.items():
        print(f"- {key}: {value}")
