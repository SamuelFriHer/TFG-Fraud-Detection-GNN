"""Downloads MLflow results from HF Hub and exports them to CSV for analysis."""

import gzip
import logging
import os
import tarfile
from pathlib import Path

import mlflow
import pandas as pd  # type: ignore

from src.utils.data_manager import DataSyncManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MLFLOW_DIR = PROJECT_ROOT / "outputs" / "mlflow"
MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"

_METRIC_COLUMNS = [
    "tags.mlflow.runName",
    "metrics.val_accuracy",
    "metrics.val_precision",
    "metrics.val_recall",
    "metrics.val_f1",
    "metrics.test_accuracy",
    "metrics.test_precision",
    "metrics.test_recall",
    "metrics.test_f1",
]

_COLUMN_RENAME = {
    "tags.mlflow.runName": "Model",
    "metrics.val_accuracy": "Val Accuracy",
    "metrics.val_precision": "Val Precision",
    "metrics.val_recall": "Val Recall",
    "metrics.val_f1": "Val F1",
    "metrics.test_accuracy": "Test Accuracy",
    "metrics.test_precision": "Test Precision",
    "metrics.test_recall": "Test Recall",
    "metrics.test_f1": "Test F1",
}


class ResultsExporter:
    """Downloads MLflow archives from HF Hub and exports experiment runs to CSV."""

    def __init__(self) -> None:
        """Initializes the exporter with the shared data sync manager."""
        self.logger = logging.getLogger(__name__)
        self.sync_manager = DataSyncManager()

    def fetch_and_export(self, experiment_name: str, hf_repo_id: str | None = None) -> Path:
        """
        Downloads the MLflow archive for the given experiment, extracts it,
        and exports all runs to a CSV file in outputs/results/.
        Returns the path to the generated CSV file.
        """
        repo_id = hf_repo_id or os.getenv("HF_MODEL_REPO_ID")
        if not repo_id:
            raise OSError("HF_MODEL_REPO_ID not set and no repo_id provided.")

        archive_name = f"mlflow_{experiment_name}.tar.gz"
        self.logger.info("Downloading '%s' from '%s'...", archive_name, repo_id)
        archive_path = self._download_archive(repo_id, archive_name)

        self.logger.info("Extracting MLflow store...")
        self._extract_archive(archive_path)

        self.logger.info("Querying MLflow experiment '%s'...", experiment_name)
        runs_frame = self._query_experiment(experiment_name)

        csv_path = self._export_to_csv(runs_frame, experiment_name)
        self.logger.info("Results exported to %s", csv_path)
        return csv_path

    def _download_archive(self, repo_id: str, archive_name: str) -> Path:
        """Downloads the MLflow tar.gz from the HF model repository."""
        local_dir = str(PROJECT_ROOT / "outputs" / "archives")
        remote_path = f"mlflow/{archive_name}"
        downloaded = self.sync_manager.download_model_file(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=local_dir,
        )
        return Path(downloaded)

    def _extract_archive(self, archive_path: Path) -> None:
        """
        Extracts the tar.gz archive into the outputs/ directory.
        If the archive is corrupted, it is deleted to allow redownload.
        """
        outputs_dir = PROJECT_ROOT / "outputs"
        self.logger.debug("Verifying and extracting archive: %s", archive_path)

        try:
            if not tarfile.is_tarfile(archive_path):
                raise tarfile.ReadError(f"Not a valid tar file: {archive_path}")

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=outputs_dir)
        except (EOFError, tarfile.ReadError, gzip.BadGzipFile) as e:
            self.logger.error("Corrupted archive detected at '%s': %s", archive_path, e)
            if archive_path.exists():
                self.logger.info("Deleting corrupted archive to allow redownload on next run.")
                archive_path.unlink()
            raise RuntimeError(
                f"Archive extraction failed for {archive_path.name}. "
                "The file was corrupted and has been deleted. Please retry the operation."
            ) from e

    def _query_experiment(self, experiment_name: str) -> pd.DataFrame:
        """Queries the local MLflow SQLite store and returns all runs as a DataFrame."""
        tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
        mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in the local MLflow store.")

        runs_frame: pd.DataFrame = mlflow.search_runs(  # type: ignore[assignment]
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_f1 DESC"],
        )

        available = [col for col in _METRIC_COLUMNS if col in runs_frame.columns]
        metric_cols = [c for c in available if c != "tags.mlflow.runName"]
        clean_frame = (
            runs_frame[available]
            .dropna(subset=metric_cols)
            .drop_duplicates(subset=["tags.mlflow.runName"], keep="first")
            .rename(columns=_COLUMN_RENAME)
        )
        return clean_frame

    def _export_to_csv(self, frame: pd.DataFrame, experiment_name: str) -> Path:
        """Writes the results DataFrame to a CSV file in outputs/results/."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / f"{experiment_name}_results.csv"
        frame.to_csv(csv_path, index=False, float_format="%.4f")
        return csv_path
