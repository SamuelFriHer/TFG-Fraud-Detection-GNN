"""Unit tests for the TraditionalPipeline class using the ZOMBIES pattern."""

from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.pipelines.experiment_config import DatasetConfig, ExperimentConfig, SplitConfig
from src.pipelines.traditional_pipeline import TraditionalPipeline


@pytest.fixture
def mock_config() -> ExperimentConfig:
    """Provides a sample typed configuration for testing."""
    return ExperimentConfig(
        dataset=DatasetConfig(handle="user/dataset", prefix="test_ds"),
        split=SplitConfig(test_size=0.4, random_state=42),
        models={"xgboost": {"n_estimators": 100}},
    )


@pytest.fixture
def pipeline(mock_config: ExperimentConfig) -> TraditionalPipeline:
    """Provides a TraditionalPipeline instance with mocked dependencies."""
    with (
        patch(
            "src.pipelines.experiment_config.ExperimentConfig.from_toml",
            return_value=mock_config,
        ),
        patch("src.pipelines.traditional_pipeline.DataSyncManager"),
        patch("src.pipelines.traditional_pipeline.DataPreprocessor"),
    ):
        return TraditionalPipeline("fake_config.toml")


class TestTraditionalPipeline:
    """Tests for the TraditionalPipeline class logic."""

    def test_simple_run_happy_path(self, pipeline: TraditionalPipeline) -> None:
        """S: Simple happy path for running the pipeline."""
        requested_models = ["xgboost"]
        mock_splits = {
            "x_train": np.array([[1]]),
            "y_train": np.array([0]),
            "x_val": np.array([[1]]),
            "y_val": np.array([0]),
            "x_test": np.array([[1]]),
            "y_test": np.array([0]),
        }

        mock_model = MagicMock()
        mock_model.evaluate.return_value = {"accuracy": 0.9}

        with (
            patch.object(pipeline, "_preprocess", return_value=mock_splits),
            patch("src.pipelines.traditional_pipeline.create_model", return_value=mock_model),
            patch("src.pipelines.traditional_pipeline.ExperimentTracker") as mock_tracker_cls,
        ):
            mock_tracker = mock_tracker_cls.return_value

            pipeline.run(requested_models)

            mock_model.train.assert_called_once()
            assert mock_model.evaluate.call_count == 2
            mock_tracker.upload_results_to_hub.assert_called_once()

    def test_zero_models_requested(self, pipeline: TraditionalPipeline) -> None:
        """Z: Running the pipeline with zero models."""
        with (
            patch.object(pipeline, "_preprocess", return_value={}),
            patch("src.pipelines.traditional_pipeline.ExperimentTracker"),
        ):
            pipeline.run([])

    def test_interface_preprocess_logic(self, pipeline: TraditionalPipeline) -> None:
        """I: Interface test for preprocessing steps."""
        cast(MagicMock, pipeline.sync_manager.download_kaggle_dataset).return_value = "path/to/data"
        cast(MagicMock, pipeline.preprocessor.load_data).return_value = pl.DataFrame(
            {"Is Laundering": [0]}
        )
        cast(MagicMock, pipeline.preprocessor.clean_data).return_value = pl.DataFrame(
            {"Is Laundering": [0]}
        )
        cast(MagicMock, pipeline.preprocessor.encode_features).return_value = pl.DataFrame(
            {"Is Laundering": [0]}
        )
        cast(MagicMock, pipeline.preprocessor.split_data).return_value = (
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
        )

        splits = pipeline._preprocess()

        assert "x_train" in splits
        cast(MagicMock, pipeline.sync_manager.download_kaggle_dataset).assert_called_once_with(
            "user/dataset"
        )
        cast(MagicMock, pipeline.preprocessor.load_data).assert_called_once()

    def test_exception_config_not_found(self) -> None:
        """E: Exception when config file does not exist."""
        with pytest.raises(FileNotFoundError, match="Config not found"):
            TraditionalPipeline("non_existent.toml")

    def test_many_models_execution(self, pipeline: TraditionalPipeline) -> None:
        """M: Running the pipeline for many models."""
        requested_models = ["model1", "model2"]
        mock_splits = {
            "x_train": np.array([[1]]),
            "y_train": np.array([0]),
            "x_val": np.array([[1]]),
            "y_val": np.array([0]),
            "x_test": np.array([[1]]),
            "y_test": np.array([0]),
        }

        mock_model = MagicMock()
        mock_model.evaluate.return_value = {}

        with (
            patch.object(pipeline, "_preprocess", return_value=mock_splits),
            patch(
                "src.pipelines.traditional_pipeline.create_model", return_value=mock_model
            ) as mock_create_model,
            patch("src.pipelines.traditional_pipeline.ExperimentTracker"),
        ):
            pipeline._train_and_evaluate(requested_models, mock_splits, MagicMock())

            assert mock_create_model.call_count == 2
            assert mock_model.train.call_count == 2

    def test_boundary_detect_categorical_columns(self, pipeline: TraditionalPipeline) -> None:
        """B: Boundary case for detecting categorical columns (exclude target)."""
        frame = pl.DataFrame(
            {
                "cat1": ["a", "b"],
                "num1": [1, 2],
                "Is Laundering": ["0", "1"],
            }
        )

        cats = pipeline._detect_categorical_columns(frame)

        assert "cat1" in cats
        assert "num1" not in cats
        assert "Is Laundering" not in cats
