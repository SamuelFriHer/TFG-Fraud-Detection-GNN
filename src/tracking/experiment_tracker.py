"""MLflow experiment tracking with Hugging Face Hub synchronization."""

import logging
import os
import tarfile
from typing import Any

import mlflow
import mlflow.pytorch
import torch

from src.utils.data_manager import DataSyncManager
from src.utils.paths import MLFLOW_DB_PATH, MLFLOW_DIR, PROJECT_ROOT


class PyTorchModelWrapper(torch.nn.Module):
    """Wraps the GNN encoder and classifier into a single PyTorch module for MLflow."""

    def __init__(self, encoder: torch.nn.Module, classifier: torch.nn.Module) -> None:
        """Initializes the wrapper with the sub-modules."""
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass for interface compatibility."""
        return None


class ExperimentTracker:
    """Wraps MLflow to track experiments locally and upload results to HF Hub."""

    def __init__(self, experiment_name: str) -> None:
        """Configures MLflow to use a local SQLite-based tracking store."""
        self.logger = logging.getLogger(__name__)
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.logger.info("MLflow experiment '%s' at %s", experiment_name, tracking_uri)

    def start_run(self, run_name: str) -> None:
        """Opens a new MLflow run with the given name."""
        mlflow.start_run(run_name=run_name)
        self.logger.info("Started MLflow run: %s", run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs hyperparameters for the active run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Logs evaluation metrics for the active run."""
        mlflow.log_metrics(metrics)
        self.logger.info("Logged metrics: %s", metrics)

    def log_model(self, model: object, model_name: str) -> None:
        """Persists the trained model, supporting sklearn, cuML, and PyTorch (GNN) backends."""
        if hasattr(model, "predict"):
            try:
                mlflow.sklearn.log_model(model, name=model_name)
            except TypeError:
                mlflow.pyfunc.log_model(name=model_name, python_model=model)
        elif isinstance(model, torch.nn.Module):
            mlflow.pytorch.log_model(model, artifact_path=model_name)
        elif (
            isinstance(model, tuple)
            and len(model) == 2
            and isinstance(model[0], torch.nn.Module)
            and isinstance(model[1], torch.nn.Module)
        ):
            wrapper = PyTorchModelWrapper(model[0], model[1])
            mlflow.pytorch.log_model(wrapper, artifact_path=model_name)
        else:
            mlflow.pyfunc.log_model(name=model_name, python_model=model)
        self.logger.info("Model artifact saved as '%s'", model_name)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Logs a local file or directory as an artifact in MLflow."""
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info("Logged artifact from %s", local_path)

    def end_run(self) -> None:
        """Closes the active MLflow run."""
        mlflow.end_run()

    def upload_results_to_hub(self, hf_repo_id: str | None = None) -> None:
        """Compresses and uploads the MLflow store to Hugging Face Hub."""
        repo_id = hf_repo_id or os.getenv("HF_MODEL_REPO_ID")
        if not repo_id:
            self.logger.warning("HF_MODEL_REPO_ID not set. Skipping MLflow upload.")
            return

        archive_name = f"mlflow_{self.experiment_name}.tar.gz"
        archive_path = PROJECT_ROOT / "outputs" / archive_name

        self.logger.info("Compressing MLflow store for upload...")
        self._compress_mlflow_store(archive_path)

        sync_manager = DataSyncManager()
        sync_manager.upload_artifact(
            local_path=str(archive_path),
            repo_id=repo_id,
            remote_filename=f"mlflow/{archive_name}",
        )
        self.logger.info("MLflow results uploaded to %s", repo_id)

    @staticmethod
    def _compress_mlflow_store(output_path: "os.PathLike[str]") -> None:
        """Creates a tar.gz archive of the MLflow tracking directory."""
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(MLFLOW_DIR, arcname="mlflow")
