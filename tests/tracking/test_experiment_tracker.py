"""Unit tests for the ExperimentTracker class."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tracking.experiment_tracker import ExperimentTracker


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, None, None]:
    """Mock the mlflow module to prevent actual tracking side effects."""
    with patch("src.tracking.experiment_tracker.mlflow") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_paths() -> Generator[dict[str, MagicMock], None, None]:
    """Mock path constants to avoid actual filesystem calls."""
    mock_mlflow_dir: MagicMock = MagicMock()
    mock_mlflow_db_path: MagicMock = MagicMock()
    mock_project_root: MagicMock = MagicMock()

    mock_mlflow_db_path.__str__.return_value = "/fake/path/mlflow.db"

    mock_outputs: MagicMock = MagicMock()
    mock_archive_path: MagicMock = MagicMock()
    mock_project_root.__truediv__.return_value = mock_outputs
    mock_outputs.__truediv__.return_value = mock_archive_path
    mock_archive_path.__str__.return_value = "/fake/outputs/mlflow_test_experiment.tar.gz"

    with (
        patch("src.tracking.experiment_tracker.MLFLOW_DIR", mock_mlflow_dir),
        patch("src.tracking.experiment_tracker.MLFLOW_DB_PATH", mock_mlflow_db_path),
        patch("src.tracking.experiment_tracker.PROJECT_ROOT", mock_project_root),
    ):
        yield {
            "mlflow_dir": mock_mlflow_dir,
            "mlflow_db_path": mock_mlflow_db_path,
            "project_root": mock_project_root,
            "archive_path": mock_archive_path,
        }


@pytest.fixture
def mock_tarfile() -> Generator[MagicMock, None, None]:
    """Mock the tarfile module to avoid creating actual tar archives."""
    with patch("src.tracking.experiment_tracker.tarfile") as mock:
        mock_tar: MagicMock = MagicMock()
        mock.open.return_value.__enter__.return_value = mock_tar
        yield mock


@pytest.fixture
def mock_data_sync_manager() -> Generator[MagicMock, None, None]:
    """Mock DataSyncManager to prevent actual Hugging Face Hub operations."""
    with patch("src.tracking.experiment_tracker.DataSyncManager") as mock_class:
        mock_instance: MagicMock = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_experiment_tracker_initialization(
    mock_mlflow: MagicMock, mock_paths: dict[str, MagicMock]
) -> None:
    """Verifies that ExperimentTracker configures mlflow correctly on init."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    assert tracker.experiment_name == "test_experiment"
    mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:////fake/path/mlflow.db")
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    mock_paths["mlflow_dir"].mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_experiment_tracker_start_run(mock_mlflow: MagicMock) -> None:
    """Verifies that start_run correctly delegates to mlflow.start_run."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.start_run(run_name="test_run")
    mock_mlflow.start_run.assert_called_once_with(run_name="test_run")


def test_experiment_tracker_log_params(mock_mlflow: MagicMock) -> None:
    """Verifies that log_params correctly delegates to mlflow.log_params."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    params: dict[str, float | int] = {"lr": 0.01, "epochs": 10}
    tracker.log_params(params)
    mock_mlflow.log_params.assert_called_once_with(params)


def test_experiment_tracker_log_metrics(mock_mlflow: MagicMock) -> None:
    """Verifies that log_metrics correctly delegates to mlflow.log_metrics with correct dict."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    metrics: dict[str, float] = {"val_loss": 0.25, "val_f1": 0.85}
    tracker.log_metrics(metrics)
    mock_mlflow.log_metrics.assert_called_once_with(metrics)


def test_experiment_tracker_end_run(mock_mlflow: MagicMock) -> None:
    """Verifies that end_run correctly delegates to mlflow.end_run."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.end_run()
    mock_mlflow.end_run.assert_called_once()


def test_experiment_tracker_log_model_sklearn(mock_mlflow: MagicMock) -> None:
    """Verifies log_model delegates to mlflow.sklearn for compatible models."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    mock_model: MagicMock = MagicMock()
    tracker.log_model(mock_model, model_name="sklearn_model")
    mock_mlflow.sklearn.log_model.assert_called_once_with(mock_model, name="sklearn_model")


def test_experiment_tracker_log_model_fallback(mock_mlflow: MagicMock) -> None:
    """Verifies log_model falls back to pyfunc if sklearn.log_model raises TypeError."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    mock_model: MagicMock = MagicMock()
    mock_mlflow.sklearn.log_model.side_effect = TypeError("Not a sklearn model")

    tracker.log_model(mock_model, model_name="other_model")

    mock_mlflow.sklearn.log_model.assert_called_once_with(mock_model, name="other_model")
    mock_mlflow.pyfunc.log_model.assert_called_once_with(
        name="other_model", python_model=mock_model
    )


def test_experiment_tracker_log_model_no_predict(mock_mlflow: MagicMock) -> None:
    """Verifies log_model delegates directly to pyfunc if model lacks 'predict'."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    mock_model: MagicMock = MagicMock(spec=[])

    tracker.log_model(mock_model, model_name="other_model")

    mock_mlflow.sklearn.log_model.assert_not_called()
    mock_mlflow.pyfunc.log_model.assert_called_once_with(
        name="other_model", python_model=mock_model
    )


def test_upload_results_to_hub_with_arg(
    mock_mlflow: MagicMock,
    mock_paths: dict[str, MagicMock],
    mock_tarfile: MagicMock,
    mock_data_sync_manager: MagicMock,
) -> None:
    """Verifies upload_results_to_hub uploads correctly when hf_repo_id is passed."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.upload_results_to_hub(hf_repo_id="test-repo-arg")

    mock_data_sync_manager.upload_artifact.assert_called_once_with(
        local_path="/fake/outputs/mlflow_test_experiment.tar.gz",
        repo_id="test-repo-arg",
        remote_filename="mlflow/mlflow_test_experiment.tar.gz",
    )


def test_upload_results_to_hub_with_env(
    mock_mlflow: MagicMock,
    mock_paths: dict[str, MagicMock],
    mock_tarfile: MagicMock,
    mock_data_sync_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies upload_results_to_hub uses environment variable when hf_repo_id is None."""
    monkeypatch.setenv("HF_MODEL_REPO_ID", "test-repo-env")
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.upload_results_to_hub(hf_repo_id=None)

    mock_data_sync_manager.upload_artifact.assert_called_once_with(
        local_path="/fake/outputs/mlflow_test_experiment.tar.gz",
        repo_id="test-repo-env",
        remote_filename="mlflow/mlflow_test_experiment.tar.gz",
    )


def test_upload_results_to_hub_missing_repo_id(
    mock_mlflow: MagicMock,
    mock_paths: dict[str, MagicMock],
    mock_tarfile: MagicMock,
    mock_data_sync_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies upload_results_to_hub skips upload if no repository ID is configured."""
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    tracker.upload_results_to_hub(hf_repo_id=None)

    mock_data_sync_manager.upload_artifact.assert_not_called()


def test_compress_mlflow_store(
    mock_mlflow: MagicMock,
    mock_paths: dict[str, MagicMock],
    mock_tarfile: MagicMock,
) -> None:
    """Verifies that _compress_mlflow_store correctly creates tar archive of MLflow dir."""
    tracker: ExperimentTracker = ExperimentTracker(experiment_name="test_experiment")
    archive_path: Path = mock_paths["archive_path"]
    tracker._compress_mlflow_store(archive_path)

    mock_tarfile.open.assert_called_once_with(archive_path, "w:gz")
    mock_tar: MagicMock = mock_tarfile.open.return_value.__enter__.return_value
    mock_tar.add.assert_called_once_with(mock_paths["mlflow_dir"], arcname="mlflow")
