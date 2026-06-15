"""Unit tests for the CLI helper functions and main entry point."""

from unittest.mock import MagicMock, patch

import pytest

from src.cli import _resolve_model_names, main


def test_resolve_model_names_valid() -> None:
    """Verifies that _resolve_model_names resolves valid models correctly."""
    result: list[str] = _resolve_model_names(["XGBoost", "RandomForest"])
    assert "XGBoost" in result
    assert "RandomForest" in result


def test_resolve_model_names_all() -> None:
    """Verifies that 'all' resolves to all registered model names."""
    result: list[str] = _resolve_model_names(["all"])
    from src.models.traditional import ALL_MODEL_NAMES

    assert result == ALL_MODEL_NAMES


def test_resolve_model_names_raises_value_error_on_unknown_model() -> None:
    """Verifies that _resolve_model_names raises ValueError for unrecognized models."""
    invalid_model_names: list[str] = ["invalid_model_name"]
    with pytest.raises(ValueError, match="Unknown model"):
        _resolve_model_names(invalid_model_names)


@patch("src.cli.ProjectLogger")
@patch("src.cli.TraditionalPipeline")
def test_cli_traditional_dispatch(
    mock_pipeline_class: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Tests the dispatch to the traditional ML pipeline with default models."""
    mock_pipeline: MagicMock = mock_pipeline_class.return_value
    test_args: list[str] = ["fraud-detect", "traditional", "--config", "config.toml"]

    with patch("sys.argv", test_args):
        main()

    mock_pipeline_class.assert_called_once_with(config_path="config.toml")
    from src.models.traditional import ALL_MODEL_NAMES

    mock_pipeline.run.assert_called_once_with(requested_models=ALL_MODEL_NAMES)


@patch("src.cli.ProjectLogger")
@patch("src.cli.TraditionalPipeline")
def test_cli_traditional_dispatch_with_models(
    mock_pipeline_class: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Tests the dispatch to traditional ML pipeline with specified models."""
    mock_pipeline: MagicMock = mock_pipeline_class.return_value
    test_args: list[str] = [
        "fraud-detect",
        "traditional",
        "--config",
        "config.toml",
        "--models",
        "XGBoost",
    ]

    with patch("sys.argv", test_args):
        main()

    mock_pipeline_class.assert_called_once_with(config_path="config.toml")
    mock_pipeline.run.assert_called_once_with(requested_models=["XGBoost"])


@patch("src.cli.ProjectLogger")
@patch("src.cli.GNNPipeline")
def test_cli_gnn_dispatch(
    mock_pipeline_class: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Tests the dispatch to the GNN pipeline."""
    mock_pipeline: MagicMock = mock_pipeline_class.return_value
    test_args: list[str] = ["fraud-detect", "gnn", "--config", "config.toml"]

    with patch("sys.argv", test_args):
        main()

    mock_pipeline_class.assert_called_once_with(config_path="config.toml")
    mock_pipeline.run.assert_called_once()


@patch("src.cli.ProjectLogger")
@patch("src.cli.GNNGridSearchPipeline")
def test_cli_gnn_grid_dispatch(
    mock_pipeline_class: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Tests the dispatch to the GNN Grid Search pipeline."""
    mock_pipeline: MagicMock = mock_pipeline_class.return_value
    test_args: list[str] = ["fraud-detect", "gnn-grid", "--config", "config.toml"]

    with patch("sys.argv", test_args):
        main()

    mock_pipeline_class.assert_called_once_with(config_path="config.toml")
    mock_pipeline.run.assert_called_once()


@patch("src.cli.ProjectLogger")
@patch("src.cli.ResultsExporter")
def test_cli_fetch_results_dispatch(
    mock_exporter_class: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Tests the dispatch to results fetching and exporting pipeline."""
    mock_exporter: MagicMock = mock_exporter_class.return_value
    test_args: list[str] = [
        "fraud-detect",
        "fetch-results",
        "--experiment",
        "exp1",
        "--repo",
        "repo1",
    ]

    with patch("sys.argv", test_args):
        main()

    mock_exporter_class.assert_called_once()
    mock_exporter.fetch_and_export.assert_called_once_with(
        experiment_name="exp1",
        hf_repo_id="repo1",
    )


@patch("src.cli.ProjectLogger")
def test_cli_missing_required_arguments(mock_logger: MagicMock) -> None:
    """Tests that missing required arguments raises SystemExit."""
    test_args: list[str] = ["fraud-detect", "traditional"]

    with patch("sys.argv", test_args), pytest.raises(SystemExit):
        main()
