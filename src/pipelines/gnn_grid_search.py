"""Pipeline to run automatic Grid Search for GNN hyperparameters."""

import gc
import itertools
import tomllib
from typing import Any

import torch
from torch_geometric.data import Data

from src.data.graph_builder import AMLGraphBuilder
from src.models.gnn.config import GNNModelConfig
from src.models.gnn.model import GNNFraudDetector
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.data_manager import DataSyncManager
from src.utils.logger import ProjectLogger


class GNNGridSearchPipeline:
    """Executes a hyperparameter search for GNN models.

    Builds the graph once and reuses it for multiple training runs
    based on a predefined hyperparameter grid, logging everything to MLflow.
    """

    def __init__(self, config_path: str) -> None:
        """Initializes by reading the base configuration."""
        self.logger = ProjectLogger.get_logger("GNNGridSearch")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.sync_manager = DataSyncManager()
        self.prefix = self.config["dataset"]["prefix"]

    def _load_graph(self) -> Data:
        """Loads and builds the graph once."""
        self.logger.info(
            "Verifying dataset %s for Grid Search", self.config["dataset"]["handle"]
        )
        dataset_dir = self.sync_manager.download_kaggle_dataset(self.config["dataset"]["handle"])

        self.logger.info("Building graph in memory for %s", self.prefix)
        builder = AMLGraphBuilder()
        test_size = float(self.config.get("split", {}).get("test_size", 0.4))
        graph = builder.build_graph(dataset_dir, self.prefix, test_size=test_size)

        self.logger.info(
            "Graph built. Nodes: %d, Edges: %d", graph.num_nodes, graph.num_edges
        )
        return graph

    def _generate_grid_combinations(self) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Generates the search space and itertools.product combinations."""
        grid = {
            "pos_weight": [3.0, 5.0, 7.0, 10.0, 12.0],
            "num_neighbors": [[10, 5], [20, 10], [30, 15]],
        }
        keys = list(grid.keys())
        values = list(grid.values())
        combinations = list(itertools.product(*values))
        return keys, combinations

    def _create_model_config(
        self,
        params: dict[str, Any],
        graph: Data,
    ) -> GNNModelConfig:
        """Creates the model configuration from fixed and variable hyperparameters."""
        base_gnn_config = self.config.get("models", {}).get("MEGA_PNA", {})
        full_params = base_gnn_config.copy()
        full_params.update(params)

        return GNNModelConfig(
            node_feat_dim=int(graph.x.size(1)),
            edge_feat_dim=int(graph.edge_attr.size(1)),
            in_channels=int(full_params["in_channels"]) if "in_channels" in full_params else None,
            hidden_channels=int(full_params["hidden_channels"]),
            num_layers=int(full_params.get("num_layers", 2)),
            lr=float(full_params["learning_rate"]),
            batch_size=int(full_params.get("batch_size", 2048)),
            epochs=int(full_params.get("epochs", 80)),
            pos_weight=float(full_params["pos_weight"]),
            dropout=float(full_params["dropout"]),
            final_dropout=float(full_params.get("final_dropout", full_params["dropout"])),
            num_neighbors=full_params["num_neighbors"],
        )

    def _train_and_evaluate(
        self,
        model: GNNFraudDetector,
        graph: Data,
        tracker: ExperimentTracker,
    ) -> None:
        """Executes training and evaluation, logging metrics and model."""
        model.train(graph)

        # Evaluate
        val_metrics = model.evaluate(graph, stage="val")
        tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        test_metrics = model.evaluate(graph, stage="test")
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Save the model for this run
        tracker.log_model(model.get_underlying_model(), model_name="model")

    def _cleanup_memory(self, model: GNNFraudDetector) -> None:
        """Frees memory occupied by the model to prevent leaks."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_single_experiment(
        self,
        idx: int,
        total_runs: int,
        params: dict[str, Any],
        graph: Data,
        tracker: ExperimentTracker,
    ) -> None:
        """Executes a single iteration of the grid search."""
        run_name = f"MEGA_PNA_Grid_{idx:03d}"
        self.logger.info("--- [Run %d/%d] Parameters: %s ---", idx, total_runs, params)

        tracker.start_run(run_name=run_name)

        # Log fixed and variable hyperparameters
        base_gnn_config = self.config.get("models", {}).get("MEGA_PNA", {})
        full_params = base_gnn_config.copy()
        full_params.update(params)
        tracker.log_params(full_params)

        try:
            model_config = self._create_model_config(params, graph)
            model = GNNFraudDetector(graph_data=graph, config=model_config)
            self._train_and_evaluate(model, graph, tracker)
        except Exception as error_exception:
            self.logger.error("Error in run %d: %s", idx, str(error_exception))
        finally:
            tracker.end_run()
            if "model" in locals():
                self._cleanup_memory(locals()["model"])

    def run(self) -> None:
        """Launches the grid search, logging each iteration as a run."""
        experiment_name = f"gnn_grid_{self.prefix}"
        tracker = ExperimentTracker(experiment_name)

        graph = self._load_graph()
        keys, combinations = self._generate_grid_combinations()
        total_runs = len(combinations)

        self.logger.info("Starting Grid Search with %d combinations", total_runs)

        for idx, combo in enumerate(combinations, start=1):
            params = dict(zip(keys, combo))
            self._run_single_experiment(
                idx=idx,
                total_runs=total_runs,
                params=params,
                graph=graph,
                tracker=tracker,
            )

        self.logger.info("Grid search completed. Uploading results to the hub...")
        tracker.upload_results_to_hub()
        self.logger.info("Process completed successfully.")
