"""Full GNN model definition for fraud detection, training and evaluation."""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import LinkNeighborLoader  # type: ignore
from torch_geometric.utils import degree  # type: ignore

from src.models.gnn.evaluator import evaluate_predictions
from src.models.gnn.layers import EdgeClassifier, MEGAPNAEncoder
from src.utils.logger import ProjectLogger


class GNNFraudDetector:
    """GNN model for transaction classification (Link Prediction) using GATv2."""

    def __init__(
        self,
        data: Data,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        batch_size: int = 2048,
        epochs: int = 10,
        aggr: str = "max",
        loss_type: str = "focal",
        alpha: float | None = None,
        gamma: float = 2.0,
        num_neighbors: list[int] | None = None,
    ) -> None:
        """Initializes components and training configurations."""
        # Calculate degree histogram for PNAConv
        train_edge_index = data.edge_index[:, data.train_mask]
        in_degree = degree(train_edge_index[1], data.num_nodes, dtype=torch.long)
        deg = torch.bincount(in_degree)

        # Inyectamos 1 dimensión más para el Ego ID
        self.encoder = MEGAPNAEncoder(
            node_feat_dim + 1, edge_feat_dim, hidden_channels, num_layers, deg=deg
        )
        self.classifier = EdgeClassifier(hidden_channels, edge_feat_dim, hidden_channels)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = 0.5
        self._aggr = aggr
        self.num_neighbors = num_neighbors or [5, 5]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.logger = ProjectLogger.get_logger("GNNFraudDetector")

    def _get_train_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader for training, using train edges for message passing and supervision."""
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_attr = data.edge_attr[data.train_mask]
        train_y = data.y[data.train_mask]

        train_data = Data(
            x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr, y=train_y
        )

        y_label = train_y.cpu()
        num_pos = int((y_label == 1).sum().item())
        num_neg = int((y_label == 0).sum().item())
        if num_pos > 0 and num_neg > 0:
            total_samples = num_pos + num_neg
            pos_w = total_samples / (2.0 * num_pos)
            neg_w = total_samples / (2.0 * num_neg)
            weights = torch.zeros(y_label.size(0), dtype=torch.float)
            weights[y_label == 1] = pos_w
            weights[y_label == 0] = neg_w
            sampler = WeightedRandomSampler(
                weights=weights.tolist(), num_samples=total_samples, replacement=True
            )
            return LinkNeighborLoader(
                train_data,
                num_neighbors=self.num_neighbors,
                edge_label_index=train_data.edge_index,
                edge_label=train_data.y,
                batch_size=self.batch_size,
                sampler=sampler,
                neg_sampling_ratio=0.0,
            )
        return LinkNeighborLoader(
            train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=train_data.edge_index,
            edge_label=train_data.y,
            batch_size=self.batch_size,
            shuffle=True,
            neg_sampling_ratio=0.0,
        )

    def _get_val_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader for validation, using train edges for message passing."""
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_attr = data.edge_attr[data.train_mask]

        val_data = Data(x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)

        val_edge_index = data.edge_index[:, data.val_mask]
        val_y = data.y[data.val_mask]

        return LinkNeighborLoader(
            val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=val_edge_index,
            edge_label=val_y,
            batch_size=self.batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0,
        )

    def _get_test_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader for testing, using train + val edges for message passing."""
        history_mask = data.train_mask | data.val_mask
        history_edge_index = data.edge_index[:, history_mask]
        history_edge_attr = data.edge_attr[history_mask]

        test_data = Data(x=data.x, edge_index=history_edge_index, edge_attr=history_edge_attr)

        test_edge_index = data.edge_index[:, data.test_mask]
        test_y = data.y[data.test_mask]

        return LinkNeighborLoader(
            test_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=test_edge_index,
            edge_label=test_y,
            batch_size=self.batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0,
        )

    def _prepare_loss_criterion(self, data: Data) -> nn.Module:
        """Prepares the loss function based on config."""
        if self.loss_type == "focal":
            alpha_val = self.alpha if self.alpha is not None else 0.5
            from src.models.gnn.loss import FocalLoss

            return FocalLoss(alpha=alpha_val, gamma=self.gamma)

        pos_weight = torch.tensor([1.0], device=self.device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _train_epoch(
        self,
        loader: LinkNeighborLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_edge_attr: torch.Tensor,
    ) -> float:
        """Trains GNN for one epoch and returns the average loss."""
        total_loss = 0.0
        total_edges = 0
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            # Ego ID Injection
            num_nodes = batch.x.size(0)
            ego_flag = torch.zeros((num_nodes, 1), device=self.device, dtype=batch.x.dtype)
            ego_flag[batch.edge_label_index[0]] = 1.0
            ego_flag[batch.edge_label_index[1]] = 1.0
            batch_x = torch.cat([batch.x, ego_flag], dim=-1)

            z = self.encoder(batch_x, batch.edge_index, batch.edge_attr)
            seed_edge_attr = train_edge_attr[batch.input_id.cpu()].to(self.device)
            out = self.classifier(z, batch.edge_label_index, seed_edge_attr)
            loss = criterion(out, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            num_batch_edges = batch.edge_label_index.size(1)
            total_loss += loss.item() * num_batch_edges
            total_edges += num_batch_edges
        return total_loss / total_edges if total_edges > 0 else 0.0

    def train(self, data: Data) -> None:
        """Trains the GNN and checkpoints the best model weights based on PR-AUC."""
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=self.lr
        )
        criterion = self._prepare_loss_criterion(data)
        loader = self._get_train_loader(data)
        train_edge_attr = data.edge_attr[data.train_mask]
        best_pr_auc = -1.0
        best_enc_state = None
        best_cls_state = None

        for epoch in range(self.epochs):
            self.encoder.train()
            self.classifier.train()
            avg_loss = self._train_epoch(loader, optimizer, criterion, train_edge_attr)
            self.logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, self.epochs, avg_loss)

            metrics = self.evaluate(data, stage="val")
            pr_auc = metrics["pr_auc"]
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_enc_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
                best_cls_state = {
                    k: v.cpu().clone() for k, v in self.classifier.state_dict().items()
                }
                self.logger.info(
                    "Best model updated at epoch %d (Val PR-AUC: %.4f)", epoch + 1, pr_auc
                )

        if best_enc_state is not None and best_cls_state is not None:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in best_enc_state.items()})
            self.classifier.load_state_dict(
                {k: v.to(self.device) for k, v in best_cls_state.items()}
            )
            self.logger.info("Restored best model weights with PR-AUC: %.4f", best_pr_auc)

    def predict(self, data: Data, stage: str = "val") -> np.ndarray:
        """Generates predictions for edges in the graph for a specific stage."""
        self.encoder.eval()
        self.classifier.eval()

        if stage == "train":
            loader = self._get_train_loader(data)
            edge_attr = data.edge_attr[data.train_mask]
        elif stage == "val":
            loader = self._get_val_loader(data)
            edge_attr = data.edge_attr[data.val_mask]
        elif stage == "test":
            loader = self._get_test_loader(data)
            edge_attr = data.edge_attr[data.test_mask]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Ego ID Injection
                num_nodes = batch.x.size(0)
                ego_flag = torch.zeros((num_nodes, 1), device=self.device, dtype=batch.x.dtype)
                ego_flag[batch.edge_label_index[0]] = 1.0
                ego_flag[batch.edge_label_index[1]] = 1.0
                batch_x = torch.cat([batch.x, ego_flag], dim=-1)

                z = self.encoder(batch_x, batch.edge_index, batch.edge_attr)
                seed_edge_attr = edge_attr[batch.input_id.cpu()].to(self.device)
                out = self.classifier(z, batch.edge_label_index, seed_edge_attr)
                preds.append(torch.sigmoid(out).cpu().numpy())
        return np.concatenate(preds)

    def evaluate(self, data: Data, stage: str = "val") -> dict[str, float]:
        """Evaluates GNN predictions, optimizing the decision threshold on validation."""
        probs = self.predict(data, stage=stage)

        if stage == "train":
            y_true = data.y[data.train_mask].cpu().numpy()
        elif stage == "val":
            y_true = data.y[data.val_mask].cpu().numpy()
        elif stage == "test":
            y_true = data.y[data.test_mask].cpu().numpy()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if stage == "val":
            metrics = evaluate_predictions(probs, y_true)
            self.threshold = metrics["optimal_threshold"]
        else:
            from sklearn.metrics import (  # type: ignore[import-untyped]
                accuracy_score,
                average_precision_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            preds = (probs > self.threshold).astype(int)
            metrics = {
                "accuracy": float(accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1_score": float(f1_score(y_true, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, probs)),
                "pr_auc": float(average_precision_score(y_true, probs)),
                "optimal_threshold": self.threshold,
            }

        self.logger.info(
            "%s Stage - Threshold: %.4f, F1: %.4f, PR-AUC: %.4f, ROC-AUC: %.4f",
            stage.capitalize(),
            self.threshold,
            metrics["f1_score"],
            metrics["pr_auc"],
            metrics["roc_auc"],
        )
        return metrics

    def get_underlying_model(self) -> object:
        """Returns the encoder and classifier tuple for serialization."""
        return (self.encoder, self.classifier)
