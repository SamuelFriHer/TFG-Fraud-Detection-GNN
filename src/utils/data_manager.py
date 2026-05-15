"""Dataset download and artifact upload via Kaggle and Hugging Face Hub."""

import logging
import os
from pathlib import Path

import kagglehub  # type: ignore
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError  # type: ignore

from src.utils.paths import PROJECT_ROOT


class DataSyncManager:
    """Manages data synchronization between local environment and Hugging Face Hub."""

    def __init__(self, token: str | None = None) -> None:
        """Initializes the data sync module and configures cache directories."""
        self.logger = logging.getLogger(__name__)
        self._configure_cache_paths()

        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            self.logger.warning("No Hugging Face token found. Functionality might be limited.")

        self.api = HfApi(token=self.token)

    @staticmethod
    def _configure_cache_paths() -> None:
        """Sets Kaggle and HF cache directories relative to the project root."""
        os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_ROOT / ".cache" / "kagglehub"))
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

    def download_dataset_file(self, repo_id: str, filename: str, local_dir: str) -> str:
        """Downloads a specific file from a standard dataset repository."""
        self.logger.info("Downloading file %s from %s...", filename, repo_id)
        try:
            downloaded_path: str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                token=self.token,
            )
            self.logger.info("Successfully downloaded to %s", downloaded_path)
            return downloaded_path
        except (HfHubHTTPError, OSError, ValueError) as download_error:
            self.logger.error("Failed to download dataset file: %s", download_error)
            raise RuntimeError(f"Download error: {download_error}") from download_error

    def download_model_file(self, repo_id: str, filename: str, local_dir: str) -> str:
        """Downloads a specific file from a model repository (e.g., MLflow archives)."""
        self.logger.info("Downloading model file %s from %s...", filename, repo_id)
        try:
            downloaded_path: str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=local_dir,
                token=self.token,
            )
            self.logger.info("Successfully downloaded to %s", downloaded_path)
            return downloaded_path
        except (HfHubHTTPError, OSError, ValueError) as download_error:
            self.logger.error("Failed to download model file: %s", download_error)
            raise RuntimeError(f"Download error: {download_error}") from download_error

    def download_kaggle_dataset(self, handle: str) -> str:
        """Downloads a dataset from Kaggle using kagglehub (auto-cached)."""
        self.logger.info("Checking/Downloading Kaggle dataset %s...", handle)
        try:
            dataset_path: str = str(kagglehub.dataset_download(handle))
            self.logger.info("Dataset ready at: %s", dataset_path)
            return dataset_path
        except (OSError, ValueError) as kaggle_error:
            self.logger.error("Failed to download from Kaggle: %s", kaggle_error)
            raise RuntimeError(f"Kaggle download error: {kaggle_error}") from kaggle_error

    def upload_artifact(self, local_path: str, repo_id: str, remote_filename: str) -> str:
        """Uploads a generated file (like a trained model) to a Hugging Face repository."""
        if not Path(local_path).exists():
            self.logger.error("Cannot upload %s: File does not exist", local_path)
            raise FileNotFoundError(f"Missing file for upload: {local_path}")

        self.logger.info("Uploading %s to %s as %s", local_path, repo_id, remote_filename)
        try:
            upload_url = self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_filename,
                repo_id=repo_id,
                repo_type="model",
            )
            self.logger.info("Successfully uploaded. Available at: %s", upload_url)
            return str(upload_url) if upload_url else ""
        except (HfHubHTTPError, OSError, ValueError) as upload_error:
            self.logger.error("Failed to upload artifact: %s", upload_error)
            raise RuntimeError(f"Upload error: {upload_error}") from upload_error
