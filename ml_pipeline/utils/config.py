from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOG_FILE = REPORTS_DIR / "pipeline.log"


def ensure_artifact_dirs() -> None:
    """Create artifact directories if they do not exist."""
    for directory in [ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
