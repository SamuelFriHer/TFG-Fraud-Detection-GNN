import logging
import os
from pathlib import Path

import kagglehub  # type: ignore
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["KAGGLEHUB_CACHE"] = str(PROJECT_ROOT / ".cache" / "kagglehub")
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")


class DataSyncManager:
    """
    Manages data synchronization between local environment and Hugging Face Hub.
    Ensures that datasets are downloaded and experiment artifacts are uploaded.
    """

    def __init__(self, token: str | None = None):
        """
        Initializes the Data Sync module connecting to Hugging Face Hub.
        Requires HF_TOKEN environment variable correctly set if token is not passed.
        """
        self.logger = logging.getLogger(__name__)

        # Load token from env if not provided
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            self.logger.warning("No Hugging Face token found. Functionality might be limited.")

        self.api = HfApi(token=self.token)

    def download_dataset_file(self, repo_id: str, filename: str, local_dir: str) -> str:
        """
        Downloads a specific file from a standard dataset repository.
        """
        self.logger.info(f"Downloading file {filename} from {repo_id}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                token=self.token,
            )
            self.logger.info(f"Successfully downloaded to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to download dataset file: {e}")
            raise RuntimeError(f"Download error: {e}") from e

    def download_model_file(self, repo_id: str, filename: str, local_dir: str) -> str:
        """Downloads a specific file from a model repository (e.g., MLflow archives)."""
        self.logger.info(f"Downloading model file {filename} from {repo_id}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=local_dir,
                token=self.token,
            )
            self.logger.info(f"Successfully downloaded to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to download model file: {e}")
            raise RuntimeError(f"Download error: {e}") from e

    def download_kaggle_dataset(self, handle: str) -> str:
        """
        Downloads a dataset from Kaggle using kagglehub.
        Kagglehub automatically handles caching so if it's downloaded, it will just return the path.
        """
        self.logger.info(f"Checking/Downloading Kaggle dataset {handle}...")
        try:
            path: str = str(kagglehub.dataset_download(handle))
            self.logger.info(f"Dataset ready at: {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to download from Kaggle: {e}")
            raise RuntimeError(f"Kaggle download error: {e}") from e

    def upload_artifact(self, local_path: str, repo_id: str, remote_filename: str) -> str:
        """
        Uploads a generated file (like a trained model) to a Hugging Face repository.
        """
        if not os.path.exists(local_path):
            self.logger.error(f"Cannot upload {local_path}: File does not exist")
            raise FileNotFoundError(f"Missing file for upload: {local_path}")

        self.logger.info(f"Uploading {local_path} to {repo_id} as {remote_filename}")
        try:
            url = self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_filename,
                repo_id=repo_id,
                repo_type="model",
            )
            self.logger.info(f"Successfully uploaded. Available at: {url}")
            # Ensure URL is returned as a string handling dict or string outputs
            return str(url) if url else ""
        except Exception as e:
            self.logger.error(f"Failed to upload artifact: {e}")
            raise RuntimeError(f"Upload error: {e}") from e
