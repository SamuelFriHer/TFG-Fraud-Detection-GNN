"""Unit tests for the GNNPipeline class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn.config import GNNModelConfig
from src.models.gnn.model import GNNFraudDetector
from src.pipelines.gnn_pipeline import GNNPipeline


@pytest.fixture
def mock_config_path(tmp_path) -> str:
    """Creates a temporary TOML configuration file for testing."""
    config_content = """
    [dataset]
    handle = "test/dataset"
    prefix = "test_prefix"

    [split]
    test_size = 0.3

    [models.GraphSAGE]
    hidden_channels = 32
    learning_rate = 0.005
    dropout = 0.2
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(config_content, encoding="utf-8")
    return str(config_file)


@pytest.fixture
def pipeline(mock_config_path: str) -> GNNPipeline:
    """Provides a GNNPipeline instance with mocked dependencies."""
    with patch("src.pipelines.gnn_pipeline.DataSyncManager"):
        return GNNPipeline(mock_config_path)


class TestGNNPipeline:
    """Unit tests for GNNPipeline logic."""

    def test_init(self, pipeline: GNNPipeline) -> None:
        """Verifies initialization and configuration loading."""
        assert pipeline.prefix == "test_prefix"
        assert pipeline.config["dataset"]["handle"] == "test/dataset"
        assert pipeline.sync_manager is not None

    def test_build_graph(self, pipeline: GNNPipeline) -> None:
        """Verifies dataset download and graph construction."""
        mock_data = MagicMock(spec=Data)
        mock_data.x = torch.rand((10, 5))
        mock_data.edge_attr = torch.rand((20, 3))

        with patch("src.pipelines.gnn_pipeline.AMLGraphBuilder") as mock_builder_cls:
            mock_builder = mock_builder_cls.return_value
            mock_builder.build_graph.return_value = mock_data

            pipeline.sync_manager.download_kaggle_dataset.return_value = "mock_dir"

            data = pipeline._build_graph()

            assert data == mock_data
            pipeline.sync_manager.download_kaggle_dataset.assert_called_once_with("test/dataset")
            mock_builder.build_graph.assert_called_once_with(
                "mock_dir", "test_prefix", test_size=0.3
            )

    def test_create_model(self, pipeline: GNNPipeline) -> None:
        """Verifies correct construction of the model and config."""
        mock_data = MagicMock(spec=Data)
        mock_data.x = torch.rand((10, 5))
        mock_data.edge_attr = torch.rand((20, 3))

        with patch("src.pipelines.gnn_pipeline.GNNFraudDetector") as mock_detector_cls:
            pipeline._create_model(mock_data)

            mock_detector_cls.assert_called_once()
            args, kwargs = mock_detector_cls.call_args
            assert kwargs["graph_data"] == mock_data
            config: GNNModelConfig = kwargs["config"]
            assert config.node_feat_dim == 5
            assert config.edge_feat_dim == 3
            assert config.hidden_channels == 32
            assert config.lr == 0.005
            assert config.dropout == 0.2

    def test_train_and_evaluate(self, pipeline: GNNPipeline) -> None:
        """Verifies training steps and metric recording."""
        mock_model = MagicMock(spec=GNNFraudDetector)
        mock_model.evaluate.side_effect = lambda data, stage: (
            {"f1": 0.8} if stage == "val" else {"f1": 0.7}
        )
        mock_model.get_underlying_model.return_value = "underlying_model"

        mock_data = MagicMock(spec=Data)
        mock_tracker = MagicMock()

        pipeline._train_and_evaluate(mock_model, mock_data, mock_tracker)

        mock_model.train.assert_called_once_with(mock_data)
        mock_model.evaluate.assert_any_call(mock_data, stage="val")
        mock_model.evaluate.assert_any_call(mock_data, stage="test")
        mock_tracker.log_metrics.assert_any_call({"val_f1": 0.8})
        mock_tracker.log_metrics.assert_any_call({"test_f1": 0.7})
        mock_tracker.log_model.assert_called_once_with(
            "underlying_model", model_name="MEGA_PNA_model"
        )

    def test_run(self, pipeline: GNNPipeline) -> None:
        """Verifies full execution sequence."""
        mock_data = MagicMock(spec=Data)
        mock_model = MagicMock(spec=GNNFraudDetector)

        with (
            patch.object(pipeline, "_build_graph", return_value=mock_data) as mock_build_graph,
            patch.object(pipeline, "_create_model", return_value=mock_model) as mock_create_model,
            patch.object(pipeline, "_train_and_evaluate") as mock_train_eval,
            patch("src.pipelines.gnn_pipeline.ExperimentTracker") as mock_tracker_cls,
        ):
            mock_tracker = mock_tracker_cls.return_value
            pipeline.run()

            mock_build_graph.assert_called_once()
            mock_create_model.assert_called_once_with(mock_data)
            mock_train_eval.assert_called_once_with(mock_model, mock_data, mock_tracker)
            mock_tracker.upload_results_to_hub.assert_called_once()
