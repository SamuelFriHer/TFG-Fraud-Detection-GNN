"""Unit tests for the ResultsExporter class using the ZOMBIES pattern."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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
            patch.object(exporter, "_extract_explainability_pngs") as mock_extract_pngs,
        ):
            # Act
            result_path = exporter.fetch_and_export(experiment_name, hf_repo_id=repo_id)

            # Assert
            assert result_path == Path("results.csv")
            mock_extract.assert_called_once_with(mock_archive)
            mock_extract_pngs.assert_called_once()

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

    def test_is_safe_path(self, exporter: ResultsExporter) -> None:
        """Tests that _is_safe_path correctly identifies safe vs unsafe paths."""
        base_dir = Path("/tmp/base")
        assert exporter._is_safe_path(base_dir, base_dir / "safe_file.txt")
        assert exporter._is_safe_path(base_dir, base_dir / "subdir" / "safe_file.txt")
        assert not exporter._is_safe_path(base_dir, base_dir / ".." / "unsafe.txt")
        assert not exporter._is_safe_path(base_dir, Path("/etc/passwd"))

    def test_safe_extractall_with_filter(self, exporter: ResultsExporter) -> None:
        """Tests that _safe_extractall uses filter='data' when supported."""
        import tarfile

        mock_tar = MagicMock(spec=tarfile.TarFile)
        with patch("tarfile.data_filter", create=True):
            exporter._safe_extractall(mock_tar, Path("/tmp/base"))
            mock_tar.extractall.assert_called_once_with(path=Path("/tmp/base"), filter="data")

    def test_safe_extractall_fallback_happy_path(self, exporter: ResultsExporter) -> None:
        """Tests manual validation fallback when PEP 706 filter is not available."""
        import tarfile

        mock_tar = MagicMock(spec=tarfile.TarFile)
        mock_member = MagicMock(spec=tarfile.TarInfo)
        mock_member.name = "safe_file.txt"
        mock_member.isreg.return_value = True
        mock_member.isdir.return_value = False
        mock_tar.getmembers.return_value = [mock_member]

        with patch("src.pipelines.results_exporter.hasattr", return_value=False):
            exporter._safe_extractall(mock_tar, Path("/tmp/base"))
            mock_tar.extractall.assert_called_once_with(path=Path("/tmp/base"))

    def test_safe_extractall_fallback_traversal(self, exporter: ResultsExporter) -> None:
        """Tests that manual validation fallback raises ExtractError on traversal."""
        import tarfile

        mock_tar = MagicMock(spec=tarfile.TarFile)
        mock_member = MagicMock(spec=tarfile.TarInfo)
        mock_member.name = "../unsafe.txt"
        mock_member.isreg.return_value = True
        mock_member.isdir.return_value = False
        mock_tar.getmembers.return_value = [mock_member]

        with (
            patch("src.pipelines.results_exporter.hasattr", return_value=False),
            pytest.raises(tarfile.ExtractError, match="Attempted path traversal"),
        ):
            exporter._safe_extractall(mock_tar, Path("/tmp/base"))

    def test_safe_extractall_fallback_unsupported_type(self, exporter: ResultsExporter) -> None:
        """Tests that fallback raises ExtractError for symlinks or special files."""
        import tarfile

        mock_tar = MagicMock(spec=tarfile.TarFile)
        mock_member = MagicMock(spec=tarfile.TarInfo)
        mock_member.name = "symlink_file.txt"
        mock_member.isreg.return_value = False
        mock_member.isdir.return_value = False
        mock_tar.getmembers.return_value = [mock_member]

        with (
            patch("src.pipelines.results_exporter.hasattr", return_value=False),
            pytest.raises(tarfile.ExtractError, match="Unsupported or unsafe member type"),
        ):
            exporter._safe_extractall(mock_tar, Path("/tmp/base"))

    def test_safe_extractall_fallback_base_dir_overwrite(self, exporter: ResultsExporter) -> None:
        """Tests that fallback raises ExtractError when trying to overwrite base dir with file."""
        import tarfile

        mock_tar = MagicMock(spec=tarfile.TarFile)
        mock_member = MagicMock(spec=tarfile.TarInfo)
        mock_member.name = "."
        mock_member.isreg.return_value = True
        mock_member.isdir.return_value = False
        mock_tar.getmembers.return_value = [mock_member]

        with (
            patch("src.pipelines.results_exporter.hasattr", return_value=False),
            pytest.raises(tarfile.ExtractError, match="Attempted to overwrite base directory"),
        ):
            exporter._safe_extractall(mock_tar, Path("/tmp/base"))

    def test_extract_archive_unsafe_deletes_file(self, exporter: ResultsExporter) -> None:
        """Tests that _extract_archive deletes the archive and raises RuntimeError on unsafe tar."""
        import tarfile

        mock_archive = MagicMock(spec=Path)
        mock_archive.name = "unsafe.tar.gz"
        mock_archive.exists.return_value = True

        with (
            patch("tarfile.is_tarfile", return_value=True),
            patch("tarfile.open") as mock_open,
        ):
            mock_tar = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_tar
            with (
                patch.object(
                    exporter, "_safe_extractall", side_effect=tarfile.ExtractError("traversal")
                ),
                pytest.raises(RuntimeError, match="Archive extraction failed"),
            ):
                exporter._extract_archive(mock_archive)

            mock_archive.unlink.assert_called_once()

    def test_extract_explainability_pngs(self, exporter: ResultsExporter, tmp_path: Path) -> None:
        """Tests that explainability PNGs are correctly identified and copied.

        They are retrieved from the mlflow directory.
        """
        mock_outputs_dir = tmp_path / "outputs"
        mlflow_dir = mock_outputs_dir / "mlflow"
        explain_src_dir = mlflow_dir / "1" / "run1" / "artifacts" / "explainability" / "XGBoost"
        explain_src_dir.mkdir(parents=True)

        dummy_png = explain_src_dir / "shap_summary.png"
        dummy_png.write_text("dummy content")

        other_dir = mlflow_dir / "1" / "run1" / "other"
        other_dir.mkdir(parents=True)
        other_png = other_dir / "ignored.png"
        other_png.write_text("ignored")

        with patch("src.pipelines.results_exporter.OUTPUTS_DIR", mock_outputs_dir):
            exporter._extract_explainability_pngs()

            dest_png = mock_outputs_dir / "explainability" / "XGBoost" / "shap_summary.png"
            assert dest_png.exists()
            assert dest_png.read_text() == "dummy content"

            ignored_dest = mock_outputs_dir / "explainability" / "ignored.png"
            assert not ignored_dest.exists()
