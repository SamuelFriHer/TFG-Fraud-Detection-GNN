"""Pipeline para ejecutar Grid Search automático de hiperparámetros de GNN."""

import gc
import itertools
import tomllib

import torch

from src.data.graph_builder import AMLGraphBuilder
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

    def run(self) -> None:
        """Lanza la búsqueda grid, registrando cada iteración como un run."""
        experiment_name = f"gnn_grid_{self.prefix}"
        tracker = ExperimentTracker(experiment_name)

        # 1. Cargar y construir el grafo una única vez
        self.logger.info(
            "Verificando dataset %s para Grid Search", self.config["dataset"]["handle"]
        )
        dataset_dir = self.sync_manager.download_kaggle_dataset(self.config["dataset"]["handle"])

        self.logger.info("Construyendo grafo en memoria para %s", self.prefix)
        builder = AMLGraphBuilder()
        test_size = float(self.config.get("split", {}).get("test_size", 0.4))
        data = builder.build_graph(dataset_dir, self.prefix, test_size=test_size)

        node_dim = data.x.size(1)
        edge_dim = data.edge_attr.size(1)
        self.logger.info("Grafo construido. Nodos: %d, Aristas: %d", data.num_nodes, data.num_edges)

        # 2. Definir el espacio de búsqueda
        grid = {
            "pos_weight": [3.0, 5.0, 7.0, 10.0, 12.0],
            "num_neighbors": [[10, 5], [20, 10], [30, 15]],
        }

        # Generar combinaciones usando itertools.product
        keys = list(grid.keys())
        values = list(grid.values())
        combinations = list(itertools.product(*values))  # type: ignore[call-overload]
        total_runs = len(combinations)

        self.logger.info("Iniciando Grid Search con %d combinaciones", total_runs)

        # 3. Iterar y entrenar
        for idx, combo in enumerate(combinations, start=1):
            params = dict(zip(keys, combo))

            run_name = f"MEGA_PNA_Grid_{idx:03d}"
            self.logger.info("--- [Run %d/%d] Parámetros: %s ---", idx, total_runs, params)

            tracker.start_run(run_name=run_name)

            # Registrar hyperparámetros fijos y variables
            base_gnn_config = self.config.get("models", {}).get("GraphSAGE", {})
            full_params = base_gnn_config.copy()
            full_params.update(params)
            tracker.log_params(full_params)

            model = GNNFraudDetector(
                data=data,
                node_feat_dim=node_dim,
                edge_feat_dim=edge_dim,
                hidden_channels=int(params["hidden_channels"]),
                num_layers=base_gnn_config.get("num_layers", 2),
                lr=float(params["learning_rate"]),
                batch_size=base_gnn_config.get("batch_size", 2048),
                epochs=base_gnn_config.get("epochs", 80),
                pos_weight=float(params["pos_weight"]),
                dropout=float(params["dropout"]),
                final_dropout=float(params["dropout"]),  # Usa el mismo dropout final
                num_neighbors=params["num_neighbors"],  # type: ignore[arg-type]
            )

            try:
                model.train(data)

                # Evaluar
                val_metrics = model.evaluate(data, stage="val")
                tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

                test_metrics = model.evaluate(data, stage="test")
                tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                # Guardar el modelo para este run
                tracker.log_model(model.get_underlying_model(), model_name="model")

            except Exception as e:
                self.logger.error("Error en run %d: %s", idx, str(e))
            finally:
                tracker.end_run()

                # 4. Forzar limpieza de memoria tras cada iteración
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.info("Grid search finalizado. Subiendo resultados al hub...")
        tracker.upload_results_to_hub()
        self.logger.info("Proceso completado exitosamente.")
