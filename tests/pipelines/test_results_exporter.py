"""Unit tests for the ResultsExporter class using the ZOMBIES pattern."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore
import pytest

from src.pipelines.results_exporter import ResultsExporter


@pytest.fixture
def exporter() -> ResultsExporter:
    """Provides a ResultsExporter instance with mocked dependencies."""
    with patch("src.pipelines.results_exporter.DataSyncManager"):
        return ResultsExporter()


class TestResultsExporter:
    """Tests for the ResultsExporter class logic."""

    def test_simple_fetch_and_export_path(self, exporter: ResultsExporter) -> None:
        """S: Simple happy path for fetching and exporting results."""
        # Arrange
        experiment_name = "test_exp"
        repo_id = "test/repo"
        mock_archive = Path("/tmp/mlflow_test_exp.tar.gz")

        mock_df = pd.DataFrame(
            {
                "tags.mlflow.runName": ["model1"],
                "metrics.val_accuracy": [0.9],
                "metrics.test_f1": [0.85],
            }
        )

        with (
            patch.object(exporter, "_download_archive", return_value=mock_archive),
            patch.object(exporter, "_extract_archive") as mock_extract,
            patch.object(exporter, "_query_experiment", return_value=mock_df),
            patch.object(exporter, "_export_to_csv", return_value=Path("results.csv")),
        ):
            # Act
            result_path = exporter.fetch_and_export(experiment_name, hf_repo_id=repo_id)

            # Assert
            assert result_path == Path("results.csv")
            mock_extract.assert_called_once_with(mock_archive)

    def test_exception_missing_repo_id(self, exporter: ResultsExporter) -> None:
        """E: Exception when no repo ID is provided or in environment."""
        with patch.dict(os.environ, {}, clear=True), patch("os.getenv", return_value=None):
            with pytest.raises(OSError, match="HF_MODEL_REPO_ID not set"):
                exporter.fetch_and_export("test_exp")

    def test_interface_mlflow_query(self, exporter: ResultsExporter) -> None:
        """I: Interface test for MLflow query logic."""
        experiment_name = "test_exp"

        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"

        mock_runs = pd.DataFrame(
            {
                "tags.mlflow.runName": ["model1", "model1", "model2"],
                "metrics.val_accuracy": [0.9, 0.8, 0.7],
                "metrics.test_f1": [0.85, 0.75, 0.65],
                "other_col": [1, 2, 3],
            }
        )

        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.get_experiment_by_name", return_value=mock_exp),
            patch("mlflow.search_runs", return_value=mock_runs),
        ):
            # Act
            result_df = exporter._query_experiment(experiment_name)

            # Assert - M: Many (handles duplicates and selects columns)
            assert len(result_df) == 2  # Duplicates dropped
            assert "Model" in result_df.columns
            assert "Val Accuracy" in result_df.columns
            assert "other_col" not in result_df.columns
            # Ensure "model1" took the first (keep="first")
            assert result_df.iloc[0]["Val Accuracy"] == 0.9

    def test_zero_runs_found(self, exporter: ResultsExporter) -> None:
        """Z: Zero runs found in experiment."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"

        # Empty DF with expected columns
        mock_runs = pd.DataFrame(columns=["tags.mlflow.runName", "metrics.val_accuracy"])

        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.get_experiment_by_name", return_value=mock_exp),
            patch("mlflow.search_runs", return_value=mock_runs),
        ):
            result_df = exporter._query_experiment("empty_exp")
            assert result_df.empty

    def test_one_run_export(self, exporter: ResultsExporter) -> None:
        """O: One run export path."""
        mock_df = pd.DataFrame({"Model": ["model1"], "Val Accuracy": [0.9]})

        with patch("src.pipelines.results_exporter.RESULTS_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()
            mock_dir.__truediv__.return_value = Path("/tmp/res.csv")

            with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
                path = exporter._export_to_csv(mock_df, "test_exp")
                assert path == Path("/tmp/res.csv")
                mock_to_csv.assert_called_once()

    def test_boundary_missing_metrics(self, exporter: ResultsExporter) -> None:
        """B: Boundary case where some metrics are missing."""
        mock_runs = pd.DataFrame(
            {
                "tags.mlflow.runName": ["good", "bad"],
                "metrics.val_accuracy": [0.9, None],
                "metrics.test_f1": [0.85, 0.8],
            }
        )

        mock_exp = MagicMock(experiment_id="1")

        with (
            patch("mlflow.get_experiment_by_name", return_value=mock_exp),
            patch("mlflow.search_runs", return_value=mock_runs),
        ):
            result_df = exporter._query_experiment("test")
            # "bad" has NaN in a metric column, should be dropped by dropna(subset=metric_cols)
            assert len(result_df) == 1
            assert result_df.iloc[0]["Model"] == "good"
