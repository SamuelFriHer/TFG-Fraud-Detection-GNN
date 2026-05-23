"""Pipeline de entrenamiento y evaluación para modelos GNN."""

import tomllib
from pathlib import Path

from src.data.graph_builder import AMLGraphBuilder
from src.models.gnn.model import GNNFraudDetector
from src.utils.logger import ProjectLogger


class GNNPipeline:
    """Orquesta la construcción del grafo, entrenamiento y evaluación de la GNN."""

    def __init__(self, config_path: str) -> None:
        """Inicializa el pipeline leyendo la configuración TOML."""
        self.logger = ProjectLogger.get_logger("GNNPipeline")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.dataset_dir = (
            Path(".cache/kagglehub/datasets") / self.config["dataset"]["handle"] / "versions/8"
        )
        self.prefix = self.config["dataset"]["prefix"]

    def run(self) -> None:
        """Ejecuta el pipeline completo."""
        self.logger.info("Iniciando construcción del grafo para %s", self.prefix)
        builder = AMLGraphBuilder()
        data = builder.build_graph(str(self.dataset_dir), self.prefix)

        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        self.logger.info("Grafo construido: %s", data)

        gnn_config = self.config["models"]["GraphSAGE"]

        self.logger.info("Instanciando GNNFraudDetector")
        model = GNNFraudDetector(
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
            hidden_channels=gnn_config["hidden_channels"],
            num_layers=gnn_config["num_layers"],
            lr=gnn_config["learning_rate"],
            batch_size=gnn_config["batch_size"],
            epochs=gnn_config["epochs"],
        )

        self.logger.info("Entrenando el modelo...")
        model.train(data)

        self.logger.info("Evaluando modelo (sobre el mismo grafo para prototipo)...")
        # Idealmente dividiríamos los edges de train/val/test en el data_builder
        metrics = model.evaluate(data)
        self.logger.info("Métricas finales: %s", metrics)
