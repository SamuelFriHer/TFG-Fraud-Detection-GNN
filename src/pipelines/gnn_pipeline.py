"""Pipeline de entrenamiento y evaluación para modelos GNN."""

import tomllib

from src.data.graph_builder import AMLGraphBuilder
from src.models.gnn.model import GNNFraudDetector
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.data_manager import DataSyncManager
from src.utils.logger import ProjectLogger


class GNNPipeline:
    """Orquesta la construcción del grafo, entrenamiento y evaluación de la GNN."""

    def __init__(self, config_path: str) -> None:
        """Inicializa el pipeline leyendo la configuración TOML."""
        self.logger = ProjectLogger.get_logger("GNNPipeline")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.sync_manager = DataSyncManager()
        self.prefix = self.config["dataset"]["prefix"]

    def run(self) -> None:
        """Ejecuta el pipeline completo, registrando parámetros y métricas en MLflow."""
        experiment_name = f"gnn_{self.prefix}"
        tracker = ExperimentTracker(experiment_name)

        self.logger.info("Descargando/Verificando dataset %s...", self.config["dataset"]["handle"])
        dataset_dir = self.sync_manager.download_kaggle_dataset(self.config["dataset"]["handle"])

        self.logger.info("Iniciando construcción del grafo para %s", self.prefix)
        builder = AMLGraphBuilder()
        test_size = float(self.config.get("split", {}).get("test_size", 0.4))
        data = builder.build_graph(dataset_dir, self.prefix, test_size=test_size)

        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        self.logger.info("Grafo construido: %s", data)

        gnn_config = self.config["models"]["GraphSAGE"]

        self.logger.info("Instanciando GNNFraudDetector (MEGA-PNA)")
        model = GNNFraudDetector(
            data=data,
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
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

        tracker.start_run(run_name="MEGA_PNA")
        tracker.log_params(gnn_config)

        self.logger.info("Entrenando el modelo...")
        model.train(data)

        self.logger.info("Evaluando sobre el split de validación...")
        val_metrics = model.evaluate(data, stage="val")
        tracker.log_metrics({f"val_{key}": value for key, value in val_metrics.items()})

        self.logger.info("Evaluando sobre el split de test...")
        test_metrics = model.evaluate(data, stage="test")
        tracker.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        tracker.log_model(model.get_underlying_model(), model_name="MEGA_PNA_model")
        tracker.end_run()

        self.logger.info("Subiendo resultados al hub...")
        tracker.upload_results_to_hub()
        self.logger.info("Pipeline de GNN completado.")
