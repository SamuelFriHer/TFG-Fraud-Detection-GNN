"""Orchestrates the full traditional ML pipeline: preprocess once, train N models."""

import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.data.preprocessor import DataPreprocessor
from src.models.traditional import create_model
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.data_manager import DataSyncManager


class TraditionalPipeline:
    """Runs traditional ML experiments from a TOML config file."""

    def __init__(self, config_path: str) -> None:
        """Loads and validates the experiment configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.sync_manager = DataSyncManager()
        self.preprocessor = DataPreprocessor()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Parses the TOML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "rb") as config_file:
            return tomllib.load(config_file)

    def run(self, requested_models: list[str]) -> None:
        """Executes the full pipeline: download → preprocess → train → evaluate → upload."""
        dataset_cfg = self.config["dataset"]
        prefix = dataset_cfg["prefix"]
        experiment_name = f"traditional_{prefix}"

        tracker = ExperimentTracker(experiment_name)

        splits = self._preprocess(dataset_cfg)
        self._train_and_evaluate(requested_models, splits, tracker)
        tracker.upload_results_to_hub()

        self.logger.info("Pipeline completed for %d model(s).", len(requested_models))

    def _preprocess(self, dataset_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
        """Downloads and preprocesses the dataset exactly once."""
        self.logger.info("Downloading dataset '%s'...", dataset_cfg["handle"])
        dataset_path = self.sync_manager.download_kaggle_dataset(dataset_cfg["handle"])

        self.logger.info("Loading and cleaning data (prefix: %s)...", dataset_cfg["prefix"])
        raw_frame = self.preprocessor.load_data(dataset_path, dataset_prefix=dataset_cfg["prefix"])
        clean_frame = self.preprocessor.clean_data(raw_frame)

        categorical_cols = self._detect_categorical_columns(clean_frame)
        self.logger.info("Encoding categorical features: %s", categorical_cols)
        encoded_frame = self.preprocessor.encode_features(clean_frame, categorical_cols)

        x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.split_data(
            encoded_frame, target_col="Is Laundering"
        )

        self.logger.info(
            "Splits — Train: %d, Val: %d, Test: %d",
            x_train.shape[0],
            x_val.shape[0],
            x_test.shape[0],
        )
        return {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def _detect_categorical_columns(self, frame: pl.DataFrame) -> list[str]:
        """Identifies string columns to encode, excluding the target."""
        categorical_cols = frame.select(pl.col(pl.Utf8)).columns
        if "Is Laundering" in categorical_cols:
            categorical_cols.remove("Is Laundering")
        return categorical_cols

    def _train_and_evaluate(
        self,
        model_names: list[str],
        splits: dict[str, np.ndarray],
        tracker: ExperimentTracker,
    ) -> None:
        """Iterates over requested models: train, evaluate on val+test, log to MLflow."""
        for model_name in model_names:
            self.logger.info("=== Running model: %s ===", model_name)
            model_params = self.config.get("models", {}).get(model_name, {})
            model = create_model(model_name, **model_params)

            tracker.start_run(run_name=model_name)
            tracker.log_params(model_params)

            model.train(splits["x_train"], splits["y_train"])

            val_metrics = model.evaluate(splits["x_val"], splits["y_val"])
            self.logger.info("Validation metrics for %s: %s", model_name, val_metrics)
            tracker.log_metrics({f"val_{key}": value for key, value in val_metrics.items()})

            test_metrics = model.evaluate(splits["x_test"], splits["y_test"])
            self.logger.info("Test metrics for %s: %s", model_name, test_metrics)
            tracker.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

            tracker.log_model(model.get_underlying_model(), model_name=f"{model_name}_model")
            tracker.end_run()
