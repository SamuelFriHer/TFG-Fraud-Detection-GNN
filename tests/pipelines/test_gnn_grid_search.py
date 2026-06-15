"""Unit tests for the GNNGridSearchPipeline class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn.config import GNNModelConfig
from src.models.gnn.model import GNNFraudDetector
from src.pipelines.gnn_grid_search import GNNGridSearchPipeline


@pytest.fixture
def mock_config_path(tmp_path) -> str:
    """Creates a temporary TOML configuration file for testing."""
    config_content = """
    [dataset]
    handle = "test/dataset"
    prefix = "test_prefix"

    [split]
    test_size = 0.3

    [models.MEGA_PNA]
    in_channels = 11
    hidden_channels = 32
    learning_rate = 0.005
    dropout = 0.2
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(config_content, encoding="utf-8")
    return str(config_file)


@pytest.fixture
def pipeline(mock_config_path: str) -> GNNGridSearchPipeline:
    """Provides a GNNGridSearchPipeline instance."""
    with patch("src.pipelines.gnn_grid_search.DataSyncManager"):
        return GNNGridSearchPipeline(mock_config_path)


class TestGNNGridSearchPipeline:
    """Unit tests for GNNGridSearchPipeline logic."""

    def test_init(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies initialization and configuration loading."""
        assert pipeline.prefix == "test_prefix"
        assert pipeline.config["dataset"]["handle"] == "test/dataset"
        assert pipeline.sync_manager is not None

    def test_load_graph(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies dataset download and graph construction."""
        mock_graph = MagicMock(spec=Data)
        mock_graph.x = torch.rand((10, 5))
        mock_graph.edge_attr = torch.rand((20, 3))
        mock_graph.num_nodes = 10
        mock_graph.num_edges = 20

        with patch("src.pipelines.gnn_grid_search.AMLGraphBuilder") as mock_builder_cls:
            mock_builder = mock_builder_cls.return_value
            mock_builder.build_graph.return_value = mock_graph

            pipeline.sync_manager.download_kaggle_dataset.return_value = "mock_dir"

            graph = pipeline._load_graph()

            assert graph == mock_graph
            pipeline.sync_manager.download_kaggle_dataset.assert_called_once_with("test/dataset")
            mock_builder.build_graph.assert_called_once_with(
                "mock_dir", "test_prefix", test_size=0.3
            )

    def test_generate_grid_combinations(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies key grid search parameter generation."""
        keys, combinations = pipeline._generate_grid_combinations()
        assert keys == ["pos_weight", "num_neighbors"]
        assert len(combinations) == 15  # 5 pos_weights * 3 num_neighbors

    def test_create_model_config(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies correct construction of the GNNModelConfig."""
        mock_graph = MagicMock(spec=Data)
        mock_graph.x = torch.rand((10, 10))
        mock_graph.edge_attr = torch.rand((20, 4))

        params = {"pos_weight": 5.0, "num_neighbors": [10, 5]}
        config = pipeline._create_model_config(params, mock_graph)

        assert isinstance(config, GNNModelConfig)
        assert config.node_feat_dim == 10
        assert config.edge_feat_dim == 4
        assert config.in_channels == 11
        assert config.hidden_channels == 32
        assert config.lr == 0.005
        assert config.dropout == 0.2
        assert config.pos_weight == 5.0
        assert config.num_neighbors == [10, 5]

    def test_train_and_evaluate(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies training steps and metric recording."""
        mock_model = MagicMock(spec=GNNFraudDetector)
        mock_model.evaluate.side_effect = lambda graph_obj, stage: (
            {"f1": 0.8} if stage == "val" else {"f1": 0.7}
        )
        mock_model.get_underlying_model.return_value = "underlying_model"

        mock_graph = MagicMock(spec=Data)
        mock_tracker = MagicMock()

        pipeline._train_and_evaluate(mock_model, mock_graph, mock_tracker)

        mock_model.train.assert_called_once_with(mock_graph)
        mock_model.evaluate.assert_any_call(mock_graph, stage="val")
        mock_model.evaluate.assert_any_call(mock_graph, stage="test")
        mock_tracker.log_metrics.assert_any_call({"val_f1": 0.8})
        mock_tracker.log_metrics.assert_any_call({"test_f1": 0.7})
        mock_tracker.log_model.assert_called_once_with("underlying_model", model_name="model")

    def test_cleanup_memory(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies deletion and garbage collection calls."""
        mock_model = MagicMock(spec=GNNFraudDetector)
        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_cuda,
        ):
            pipeline._cleanup_memory(mock_model)
            mock_gc.assert_called_once()
            mock_cuda.assert_called_once()

    def test_run_single_experiment(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies that a single run proceeds, logs params, and handles errors."""
        mock_graph = MagicMock(spec=Data)
        mock_tracker = MagicMock()
        params = {"pos_weight": 5.0, "num_neighbors": [10, 5]}

        with (
            patch.object(pipeline, "_create_model_config") as mock_create_config,
            patch.object(pipeline, "_train_and_evaluate") as mock_train_eval,
            patch.object(pipeline, "_cleanup_memory") as mock_cleanup,
            patch("src.pipelines.gnn_grid_search.GNNFraudDetector") as mock_detector_cls,
        ):
            mock_detector = mock_detector_cls.return_value
            pipeline._run_single_experiment(
                idx=1,
                total_runs=1,
                params=params,
                graph=mock_graph,
                tracker=mock_tracker,
            )

            mock_tracker.start_run.assert_called_once_with(run_name="MEGA_PNA_Grid_001")
            mock_tracker.log_params.assert_called_once()
            mock_create_config.assert_called_once_with(
                {"pos_weight": 5.0, "num_neighbors": [10, 5]}, mock_graph
            )
            mock_detector_cls.assert_called_once_with(
                graph_data=mock_graph, config=mock_create_config.return_value
            )
            mock_train_eval.assert_called_once_with(mock_detector, mock_graph, mock_tracker)
            mock_tracker.end_run.assert_called_once()
            mock_cleanup.assert_called_once_with(mock_detector)

    def test_run_single_experiment_exception(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies exceptions during experiment run are logged but do not crash the pipeline."""
        mock_graph = MagicMock(spec=Data)
        mock_tracker = MagicMock()
        params = {"pos_weight": 5.0, "num_neighbors": [10, 5]}

        with (
            patch.object(
                pipeline, "_create_model_config", side_effect=Exception("Model config error")
            ),
            patch.object(pipeline, "_cleanup_memory") as mock_cleanup,
        ):
            # Should not raise exception
            pipeline._run_single_experiment(
                idx=1,
                total_runs=1,
                params=params,
                graph=mock_graph,
                tracker=mock_tracker,
            )
            mock_tracker.end_run.assert_called_once()
            # Cleanup should not be called since model creation failed
            mock_cleanup.assert_not_called()

    def test_run(self, pipeline: GNNGridSearchPipeline) -> None:
        """Verifies full grid execution sequence."""
        mock_graph = MagicMock(spec=Data)

        with (
            patch.object(pipeline, "_load_graph", return_value=mock_graph),
            patch.object(
                pipeline, "_generate_grid_combinations", return_value=(["p"], [(1.0,), (2.0,)])
            ),
            patch.object(pipeline, "_run_single_experiment") as mock_run_exp,
            patch("src.pipelines.gnn_grid_search.ExperimentTracker") as mock_tracker_cls,
        ):
            mock_tracker = mock_tracker_cls.return_value
            pipeline.run()

            assert mock_run_exp.call_count == 2
            mock_tracker.upload_results_to_hub.assert_called_once()
