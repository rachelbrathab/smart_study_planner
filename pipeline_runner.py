from ml_pipeline.step_01_goal import BusinessGoalStep
from ml_pipeline.step_02_task_type import MLTaskTypeStep
from ml_pipeline.step_03_target import TargetLabelStep
from ml_pipeline.step_04_metrics import MetricsStep
from ml_pipeline.step_05_data_source import DataSourceStep
from ml_pipeline.step_06_ingestion import DataIngestionStep
from ml_pipeline.step_07_eda import EDAStep
from ml_pipeline.step_08_data_quality import DataQualityStep
from ml_pipeline.step_09_label_quality import LabelQualityStep
from ml_pipeline.step_10_split import DataSplitStep
from ml_pipeline.step_11_cleaning import DataCleaningStep
from ml_pipeline.step_12_missing import MissingValuesStep
from ml_pipeline.step_13_outliers import OutlierStep
from ml_pipeline.step_14_feature_engineering import FeatureEngineeringStep
from ml_pipeline.step_15_transformation import TransformationStep
from ml_pipeline.step_16_baseline import BaselineModelStep
from ml_pipeline.step_17_training import MainModelTrainingStep
from ml_pipeline.step_18_tuning import HyperparameterTuningStep
from ml_pipeline.step_19_evaluation import EvaluationStep
from ml_pipeline.step_20_deployment import DeploymentStep
from ml_pipeline.utils.logger import get_logger


def run_pipeline() -> None:
    logger = get_logger("ml_pipeline.runner")

    steps = [
        ("Step 1", BusinessGoalStep()),
        ("Step 2", MLTaskTypeStep()),
        ("Step 3", TargetLabelStep()),
        ("Step 4", MetricsStep()),
        ("Step 5", DataSourceStep()),
        ("Step 6", DataIngestionStep()),
        ("Step 7", EDAStep()),
        ("Step 8", DataQualityStep()),
        ("Step 9", LabelQualityStep()),
        ("Step 10", DataSplitStep()),
        ("Step 11", DataCleaningStep()),
        ("Step 12", MissingValuesStep()),
        ("Step 13", OutlierStep()),
        ("Step 14", FeatureEngineeringStep()),
        ("Step 15", TransformationStep()),
        ("Step 16", BaselineModelStep()),
        ("Step 17", MainModelTrainingStep()),
        ("Step 18", HyperparameterTuningStep()),
        ("Step 19", EvaluationStep()),
        ("Step 20", DeploymentStep()),
    ]

    logger.info("Starting 20-step ML pipeline run")

    for label, step in steps:
        logger.info("Running %s", label)
        try:
            step.run()
        except Exception:
            logger.exception("Pipeline stopped due to failure in %s", label)
            raise

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
