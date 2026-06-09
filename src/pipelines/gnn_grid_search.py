"""Pipeline para ejecutar Grid Search automático de hiperparámetros de GNN."""

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
    """Ejecuta una búsqueda de hiperparámetros para modelos GNN.

    Construye el grafo una sola vez y lo reutiliza para múltiples entrenamientos
    basados en un grid de hiperparámetros predefinido, registrando todo en MLflow.
    """

    def __init__(self, config_path: str) -> None:
        """Inicializa leyendo la configuración base."""
        self.logger = ProjectLogger.get_logger("GNNGridSearch")
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.sync_manager = DataSyncManager()
        self.prefix = self.config["dataset"]["prefix"]

    def _load_graph_data(self) -> tuple[Data, int, int]:
        """Carga y construye el grafo una única vez."""
        self.logger.info(
            "Verificando dataset %s para Grid Search", self.config["dataset"]["handle"]
        )
        dataset_dir = self.sync_manager.download_kaggle_dataset(self.config["dataset"]["handle"])

        self.logger.info("Construyendo grafo en memoria para %s", self.prefix)
        builder = AMLGraphBuilder()
        test_size = float(self.config.get("split", {}).get("test_size", 0.4))
        data = builder.build_graph(dataset_dir, self.prefix, test_size=test_size)

        node_dim = int(data.x.size(1))
        edge_dim = int(data.edge_attr.size(1))
        self.logger.info("Grafo construido. Nodos: %d, Aristas: %d", data.num_nodes, data.num_edges)
        return data, node_dim, edge_dim

    def _generate_grid_combinations(self) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Genera el espacio de búsqueda e itertools.product de combinaciones."""
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
        node_feat_dim: int,
        edge_feat_dim: int,
    ) -> GNNModelConfig:
        """Crea la configuración del modelo a partir de hyperparámetros fijos y variables."""
        base_gnn_config = self.config.get("models", {}).get("MEGA_PNA", {})
        full_params = base_gnn_config.copy()
        full_params.update(params)

        return GNNModelConfig(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
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
        data: Data,
        tracker: ExperimentTracker,
    ) -> None:
        """Ejecuta el entrenamiento y evaluación del modelo, registrando métricas y modelo."""
        model.train(data)

        # Evaluar
        val_metrics = model.evaluate(data, stage="val")
        tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        test_metrics = model.evaluate(data, stage="test")
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Guardar el modelo para este run
        tracker.log_model(model.get_underlying_model(), model_name="model")

    def _cleanup_memory(self, model: GNNFraudDetector) -> None:
        """Libera la memoria ocupada por el modelo para evitar fugas."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_single_experiment(
        self,
        idx: int,
        total_runs: int,
        keys: list[str],
        combo: tuple[Any, ...],
        data: Data,
        node_dim: int,
        edge_dim: int,
        tracker: ExperimentTracker,
    ) -> None:
        """Ejecuta una iteración individual del grid search."""
        params = dict(zip(keys, combo))
        run_name = f"MEGA_PNA_Grid_{idx:03d}"
        self.logger.info("--- [Run %d/%d] Parámetros: %s ---", idx, total_runs, params)

        tracker.start_run(run_name=run_name)

        # Registrar hyperparámetros fijos y variables
        base_gnn_config = self.config.get("models", {}).get("MEGA_PNA", {})
        full_params = base_gnn_config.copy()
        full_params.update(params)
        tracker.log_params(full_params)

        try:
            model_config = self._create_model_config(params, node_dim, edge_dim)
            model = GNNFraudDetector(graph_data=data, config=model_config)
            self._train_and_evaluate(model, data, tracker)
        except Exception as e:
            self.logger.error("Error en run %d: %s", idx, str(e))
        finally:
            tracker.end_run()
            if "model" in locals():
                self._cleanup_memory(locals()["model"])

    def run(self) -> None:
        """Lanza la búsqueda grid, registrando cada iteración como un run."""
        experiment_name = f"gnn_grid_{self.prefix}"
        tracker = ExperimentTracker(experiment_name)

        data, node_dim, edge_dim = self._load_graph_data()
        keys, combinations = self._generate_grid_combinations()
        total_runs = len(combinations)

        self.logger.info("Iniciando Grid Search con %d combinaciones", total_runs)

        for idx, combo in enumerate(combinations, start=1):
            self._run_single_experiment(
                idx, total_runs, keys, combo, data, node_dim, edge_dim, tracker
            )

        self.logger.info("Grid search finalizado. Subiendo resultados al hub...")
        tracker.upload_results_to_hub()
        self.logger.info("Proceso completado exitosamente.")
