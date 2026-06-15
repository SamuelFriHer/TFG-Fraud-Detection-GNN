"""Orchestrates the full traditional ML pipeline: preprocess once, train N models."""

import logging
import time

import numpy as np
import polars as pl

from src.config.experiment_config import ExperimentConfig
from src.data.preprocessor import DataPreprocessor
from src.models.interfaces import ITraditionalModel
from src.models.traditional import create_model
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.data_manager import DataSyncManager


class TraditionalPipeline:
    """Runs traditional ML experiments from a TOML config file."""

    def __init__(self, config_path: str) -> None:
        """Loads and validates the experiment configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = ExperimentConfig.from_toml(config_path)
        self.sync_manager = DataSyncManager()
        self.preprocessor = DataPreprocessor()

    def run(self, requested_models: list[str]) -> None:
        """Executes the full pipeline: download → preprocess → train → evaluate → upload."""
        experiment_name = f"traditional_{self.config.dataset.prefix}"

        tracker = ExperimentTracker(experiment_name)

        splits = self._preprocess()
        self._train_and_evaluate(requested_models, splits, tracker)
        tracker.upload_results_to_hub()

        self.logger.info("Pipeline completed for %d model(s).", len(requested_models))

    def _preprocess(self) -> dict[str, np.ndarray]:
        """Downloads and preprocesses the dataset exactly once."""
        handle = self.config.dataset.handle
        self.logger.info("Downloading dataset '%s'...", handle)
        dataset_path = self.sync_manager.download_kaggle_dataset(handle)
        encoded_frame = self._get_encoded_frame(dataset_path)
        return self._split_and_format_data(encoded_frame)

    def _get_encoded_frame(self, dataset_path: str) -> pl.DataFrame:
        """Loads, cleans, and encodes features from the dataset path."""
        prefix = self.config.dataset.prefix
        self.logger.info("Loading and cleaning data (prefix: %s)...", prefix)
        raw_frame = self.preprocessor.load_data(dataset_path, dataset_prefix=prefix)
        clean_frame = self.preprocessor.clean_data(raw_frame)
        categorical_cols = self._detect_categorical_columns(clean_frame)
        self.logger.info("Encoding categorical features: %s", categorical_cols)
        return self.preprocessor.encode_features(clean_frame, categorical_cols)

    def _split_and_format_data(self, encoded_frame: pl.DataFrame) -> dict[str, np.ndarray]:
        """Splits the encoded frame and returns a dictionary of dataset splits."""
        split_cfg = self.config.split
        x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.split_data(
            encoded_frame,
            target_col="Is Laundering",
            test_size=split_cfg.test_size,
            random_state=split_cfg.random_state,
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

    @staticmethod
    def _detect_categorical_columns(frame: pl.DataFrame) -> list[str]:
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
            self._run_model_lifecycle(model_name, splits, tracker)

    def _run_model_lifecycle(
        self,
        model_name: str,
        splits: dict[str, np.ndarray],
        tracker: ExperimentTracker,
    ) -> None:
        """Trains, evaluates, and logs a single model's metrics and state."""
        self.logger.info("=== Running model: %s ===", model_name)
        model_params = self.config.models.get(model_name, {})
        model = create_model(model_name, **model_params)

        tracker.start_run(run_name=model_name)
        tracker.log_params(model_params)

        self._train_model(model, model_name, splits["x_train"], splits["y_train"])
        self._evaluate_and_log(model, model_name, splits, tracker)

        tracker.log_model(model.get_underlying_model(), model_name=f"{model_name}_model")
        tracker.end_run()

    def _train_model(
        self,
        model: ITraditionalModel,
        model_name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Trains the model and logs the training duration."""
        self.logger.info(
            "Training %s on %d samples with %d features...",
            model_name,
            x_train.shape[0],
            x_train.shape[1],
        )
        start_time = time.perf_counter()
        model.train(x_train, y_train)
        duration = time.perf_counter() - start_time
        self.logger.info("Training of %s completed in %.2f seconds.", model_name, duration)

    def _evaluate_and_log(
        self,
        model: ITraditionalModel,
        model_name: str,
        splits: dict[str, np.ndarray],
        tracker: ExperimentTracker,
    ) -> None:
        """Evaluates the model on validation/test sets and logs the metrics."""
        val_metrics = model.evaluate(splits["x_val"], splits["y_val"])
        self.logger.info("Validation metrics for %s: %s", model_name, val_metrics)
        tracker.log_metrics({f"val_{key}": value for key, value in val_metrics.items()})

        test_metrics = model.evaluate(splits["x_test"], splits["y_test"])
        self.logger.info("Test metrics for %s: %s", model_name, test_metrics)
        tracker.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
