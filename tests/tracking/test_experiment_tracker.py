"""Unit tests for the ExperimentTracker class."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.tracking.experiment_tracker import ExperimentTracker


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, None, None]:
    """Mock the mlflow module to prevent actual tracking side effects."""
    with patch("src.tracking.experiment_tracker.mlflow") as mock:
        yield mock


def test_experiment_tracker_initialization(mock_mlflow: MagicMock) -> None:
    """Verifies that ExperimentTracker configures mlflow correctly on init."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    assert tracker.experiment_name == "test_experiment"
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")


def test_experiment_tracker_start_run(mock_mlflow: MagicMock) -> None:
    """Verifies that start_run correctly delegates to mlflow.start_run."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.start_run(run_name="test_run")
    mock_mlflow.start_run.assert_called_once_with(run_name="test_run")


def test_experiment_tracker_log_params(mock_mlflow: MagicMock) -> None:
    """Verifies that log_params correctly delegates to mlflow.log_params."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    params = {"lr": 0.01, "epochs": 10}
    tracker.log_params(params)
    mock_mlflow.log_params.assert_called_once_with(params)


def test_experiment_tracker_log_metrics(mock_mlflow: MagicMock) -> None:
    """Verifies that log_metrics correctly delegates to mlflow.log_metrics with correct dict."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    metrics = {"val_loss": 0.25, "val_f1": 0.85}
    tracker.log_metrics(metrics)
    mock_mlflow.log_metrics.assert_called_once_with(metrics)


def test_experiment_tracker_end_run(mock_mlflow: MagicMock) -> None:
    """Verifies that end_run correctly delegates to mlflow.end_run."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.end_run()
    mock_mlflow.end_run.assert_called_once()


def test_experiment_tracker_log_model_sklearn(mock_mlflow: MagicMock) -> None:
    """Verifies log_model delegates to mlflow.sklearn for compatible models."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    mock_model = MagicMock()
    tracker.log_model(mock_model, model_name="sklearn_model")
    mock_mlflow.sklearn.log_model.assert_called_once_with(mock_model, name="sklearn_model")


def test_experiment_tracker_log_model_fallback(mock_mlflow: MagicMock) -> None:
    """Verifies log_model falls back to pyfunc if sklearn.log_model raises TypeError."""
    tracker = ExperimentTracker(experiment_name="test_experiment")
    mock_model = MagicMock()
    mock_mlflow.sklearn.log_model.side_effect = TypeError("Not a sklearn model")

    tracker.log_model(mock_model, model_name="other_model")

    mock_mlflow.sklearn.log_model.assert_called_once_with(mock_model, name="other_model")
    mock_mlflow.pyfunc.log_model.assert_called_once_with(
        name="other_model", python_model=mock_model
    )
