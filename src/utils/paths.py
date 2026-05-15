"""Centralized path resolution for the project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MLFLOW_DIR: Path = OUTPUTS_DIR / "mlflow"
MLFLOW_DB_PATH: Path = MLFLOW_DIR / "mlflow.db"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"
ARCHIVES_DIR: Path = OUTPUTS_DIR / "archives"
