"""Training and evaluation pipeline for GNN models."""

import tomllib

from torch_geometric.data import Data

from src.data.graph_builder import AMLGraphBuilder
from src.explainability.gnn_explainer import GnnExplainerModel
from src.explainability.visualizers import GnnVisualizer
from src.models.gnn.config import GNNModelConfig
from src.models.gnn.model import GNNFraudDetector
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.data_manager import DataSyncManager
from src.utils.logger import ProjectLogger


class GNNPipeline:
    """Orchestrates graph construction, training, and evaluation for GNN models."""

    def __init__(self, config_path: str) -> None:
        """Initializes the pipeline by loading the TOML configuration."""
        self.logger = ProjectLogger.get_logger("GNNPipeline")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.sync_manager = DataSyncManager()
        self.prefix: str = self.config["dataset"]["prefix"]

    def _build_graph(self) -> Data:
        """Downloads the dataset and builds the graph representation."""
        self.logger.info("Downloading/Verifying dataset %s...", self.config["dataset"]["handle"])
        dataset_dir = self.sync_manager.download_kaggle_dataset(self.config["dataset"]["handle"])

        self.logger.info("Starting graph construction for %s", self.prefix)
        builder = AMLGraphBuilder()
        test_size = float(self.config.get("split", {}).get("test_size", 0.4))
        data = builder.build_graph(dataset_dir, self.prefix, test_size=test_size)
        self.logger.info("Graph built: %s", data)
        return data

    def _create_model(self, data: Data) -> GNNFraudDetector:
        """Configures and initializes the GNN model."""
        node_dim: int = data.x.size(1)
        edge_dim: int = data.edge_attr.size(1)
        gnn_config: dict = self.config["models"]["MEGA_PNA"]

        self.logger.info("Instantiating GNNFraudDetector (MEGA-PNA)")
        model_config = GNNModelConfig(
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
            in_channels=gnn_config.get("in_channels"),
            hidden_channels=gnn_config.get("hidden_channels", 64),
            num_layers=gnn_config.get("num_layers", 2),
            lr=gnn_config.get("learning_rate", 0.001),
            batch_size=gnn_config.get("batch_size", 2048),
            epochs=gnn_config.get("epochs", 80),
            pos_weight=gnn_config.get("pos_weight", None),
            dropout=gnn_config.get("dropout", 0.1),
            final_dropout=gnn_config.get("final_dropout", 0.1),
            num_neighbors=gnn_config.get("num_neighbors", [20, 10]),
        )
        return GNNFraudDetector(
            graph_data=data,
            config=model_config,
        )

    def _train_and_evaluate(
        self,
        model: GNNFraudDetector,
        data: Data,
        tracker: ExperimentTracker,
    ) -> None:
        """Trains the model and evaluates it on validation and test splits."""
        gnn_config: dict = self.config["models"]["MEGA_PNA"]
        tracker.start_run(run_name="MEGA_PNA")
        tracker.log_params(gnn_config)

        self.logger.info("Training the model...")
        model.train(data)

        for stage in ["val", "test"]:
            self.logger.info("Evaluating on %s split...", stage)
            metrics = model.evaluate(data, stage=stage)
            tracker.log_metrics({f"{stage}_{key}": value for key, value in metrics.items()})

        tracker.log_model(model.get_underlying_model(), model_name="MEGA_PNA_model")

        self._explain_and_log(model, data, tracker)

        tracker.end_run()

    def _explain_and_log(
        self,
        model: GNNFraudDetector,
        data: Data,
        tracker: ExperimentTracker,
    ) -> None:
        """Explains a subset of predictions and logs visualizations."""
        from src.utils.paths import PROJECT_ROOT

        self.logger.info("Generating GNN explanations...")
        explainer = GnnExplainerModel(model)

        target_edge_idx = 0
        if hasattr(data, "y") and data.y is not None:
            pos_edges = (data.y == 1).nonzero(as_tuple=True)[0]
            if len(pos_edges) > 0:
                target_edge_idx = int(pos_edges[0].item())

        explanation = explainer.explain_graph(data, index=target_edge_idx)

        output_dir = PROJECT_ROOT / "outputs" / "explainability" / "gnn"
        output_dir.mkdir(parents=True, exist_ok=True)

        subgraph_path = str(output_dir / f"gnn_subgraph_edge_{target_edge_idx}.png")

        GnnVisualizer.save_subgraph_plot(explanation, data, subgraph_path)
        tracker.log_artifact(subgraph_path, "explainability/gnn")

    def run(self) -> None:
        """Runs the complete GNN training and evaluation pipeline."""
        experiment_name = f"gnn_{self.prefix}"
        tracker = ExperimentTracker(experiment_name)

        data = self._build_graph()
        model = self._create_model(data)

        self._train_and_evaluate(model, data, tracker)

        self.logger.info("Uploading results to the hub...")
        tracker.upload_results_to_hub()
        self.logger.info("GNN pipeline completed.")
