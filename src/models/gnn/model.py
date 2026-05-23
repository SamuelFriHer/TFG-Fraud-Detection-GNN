"""Definición del modelo GNN completo para detección de fraude y su entrenamiento."""

import numpy as np
import torch
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import LinkNeighborLoader  # type: ignore

from src.models.gnn.layers import EdgeClassifier, GraphSAGEEncoder
from src.utils.logger import ProjectLogger


class GNNFraudDetector:
    """Modelo GNN ensamblado para clasificación de transacciones (Link Prediction)."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        batch_size: int = 2048,
        epochs: int = 10,
    ) -> None:
        """Inicializa los componentes de la red y configuración de entrenamiento."""
        self.encoder = GraphSAGEEncoder(node_feat_dim, hidden_channels, num_layers)
        self.classifier = EdgeClassifier(hidden_channels, edge_feat_dim, hidden_channels)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = 0.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.classifier.to(self.device)

        self.logger = ProjectLogger.get_logger("GNNFraudDetector")

    def _get_loader(self, data: Data, is_train: bool) -> LinkNeighborLoader:
        """Construye un dataloader para muestreo de enlaces y vecinos."""
        # Se asume que data ya fue dividida o estamos entrenando sobre el grafo entero
        return LinkNeighborLoader(
            data,
            num_neighbors=[10] * len(self.encoder.convs),
            edge_label_index=data.edge_index,
            edge_label=data.y,
            batch_size=self.batch_size,
            shuffle=is_train,
            neg_sampling_ratio=0.0,
        )

    def train(self, data: Data) -> None:
        """Entrena la GNN utilizando el grafo proporcionado."""
        self.encoder.train()
        self.classifier.train()

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=self.lr
        )

        # Calcular pos_weight dinámicamente para manejar desbalanceo de clases
        num_pos = float((data.y == 1).sum().item())
        num_neg = float((data.y == 0).sum().item())
        if num_pos > 0:
            weight_val = num_neg / num_pos
            pos_weight = torch.tensor([weight_val], device=self.device)
            self.logger.info("Weight for positive class: %.4f", weight_val)
        else:
            pos_weight = torch.tensor([1.0], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loader = self._get_loader(data, is_train=True)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                z = self.encoder(batch.x, batch.edge_index)
                seed_edge_attr = data.edge_attr[batch.input_id].to(self.device)
                out = self.classifier(z, batch.edge_label_index, seed_edge_attr)

                loss = criterion(out, batch.edge_label.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.edge_label_index.size(1)

            avg_loss = total_loss / data.edge_index.size(1)
            self.logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, self.epochs, avg_loss)

    def predict(self, data: Data) -> np.ndarray:
        """Genera predicciones sobre las aristas del grafo."""
        self.encoder.eval()
        self.classifier.eval()
        loader = self._get_loader(data, is_train=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                z = self.encoder(batch.x, batch.edge_index)
                seed_edge_attr = data.edge_attr[batch.input_id].to(self.device)
                out = self.classifier(z, batch.edge_label_index, seed_edge_attr)
                probs = torch.sigmoid(out)
                preds.append(probs.cpu().numpy())

        return np.concatenate(preds)

    def _find_optimal_threshold(
        self, y_true: np.ndarray, probs: np.ndarray
    ) -> tuple[float, float]:
        """Finds the decision threshold that maximizes the F1-Score.

        Args:
            y_true: Ground truth binary labels.
            probs: Predicted probabilities.

        Returns:
            A tuple containing the optimal threshold and the best F1-Score.
        """
        precisions: np.ndarray
        recalls: np.ndarray
        thresholds: np.ndarray
        precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
        f1_scores: np.ndarray = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx: int = int(np.argmax(f1_scores))
        best_threshold: float = (
            float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        )
        best_f1: float = float(f1_scores[best_idx])
        return best_threshold, best_f1

    def evaluate(self, data: Data) -> dict[str, float]:
        """Evaluates predictions on the graph, optimizing the threshold for F1-Score."""
        probs: np.ndarray = self.predict(data)
        y_true: np.ndarray = data.y.cpu().numpy()

        optimal_threshold: float
        best_f1: float
        optimal_threshold, best_f1 = self._find_optimal_threshold(y_true, probs)
        self.threshold = optimal_threshold
        self.logger.info("Optimal threshold found: %.4f (Best F1: %.4f)", self.threshold, best_f1)

        preds: np.ndarray = (probs > self.threshold).astype(int)

        accuracy: float = float(accuracy_score(y_true, preds))
        precision: float = float(precision_score(y_true, preds, zero_division=0))
        recall: float = float(recall_score(y_true, preds, zero_division=0))
        f1: float = float(f1_score(y_true, preds, zero_division=0))
        roc_auc: float = float(roc_auc_score(y_true, probs))
        pr_auc: float = float(average_precision_score(y_true, probs))

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": self.threshold,
        }
        return metrics

    def get_underlying_model(self) -> object:
        """Devuelve una tupla con los modelos para su serialización."""
        return (self.encoder, self.classifier)
