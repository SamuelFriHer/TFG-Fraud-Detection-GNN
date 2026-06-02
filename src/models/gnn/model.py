"""Full GNN model definition for fraud detection, training and evaluation."""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree

from src.models.gnn.evaluator import evaluate_predictions
from src.models.gnn.layers import EdgeClassifier, MEGAPNAEncoder
from src.models.gnn.loss import build_weighted_bce_loss, compute_pos_weight
from src.utils.logger import ProjectLogger

MAX_GRAD_NORM = 1.0
SCHEDULER_MIN_LR = 1e-6


class GNNFraudDetector:
    """GNN model for edge-level transaction classification using MEGA-PNA."""

    def __init__(
        self,
        data: Data,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        batch_size: int = 2048,
        epochs: int = 80,
        pos_weight: float | None = None,
        dropout: float = 0.1,
        final_dropout: float = 0.1,
        num_neighbors: list[int] | None = None,
    ) -> None:
        """Initializes MEGA-PNA encoder, edge classifier and training config."""
        deg = self._compute_degree_histogram(data)

        self.encoder = MEGAPNAEncoder(
            node_feat_dim + 1,
            edge_feat_dim,
            hidden_channels,
            num_layers,
            deg=deg,
            dropout=dropout,
        )
        self.classifier = EdgeClassifier(
            hidden_channels,
            edge_feat_dim,
            hidden_channels,
            final_dropout=final_dropout,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.threshold = 0.5
        self.num_neighbors = num_neighbors or [20, 10]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.logger = ProjectLogger.get_logger("GNNFraudDetector")

    def _compute_degree_histogram(self, data: Data) -> torch.Tensor:
        """Computes in-degree histogram from training edges for PNAConv."""
        train_edge_index = data.edge_index[:, data.train_mask]
        in_degree = degree(train_edge_index[1], data.num_nodes, dtype=torch.long)
        return torch.bincount(in_degree)

    def _get_train_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader with weighted sampling for class-balanced training."""
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_attr = data.edge_attr[data.train_mask]
        train_y = data.y[data.train_mask]

        train_data = Data(
            x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr, y=train_y
        )

        sampler = self._build_weighted_sampler(train_y)
        if sampler is not None:
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

    def _build_weighted_sampler(self, labels: torch.Tensor) -> WeightedRandomSampler | None:
        """Builds a WeightedRandomSampler for class-balanced mini-batches."""
        y_cpu = labels.cpu()
        num_pos = int((y_cpu == 1).sum().item())
        num_neg = int((y_cpu == 0).sum().item())
        if num_pos == 0 or num_neg == 0:
            return None

        total_samples = num_pos + num_neg
        pos_w = total_samples / (2.0 * num_pos)
        neg_w = total_samples / (2.0 * num_neg)
        weights = torch.where(y_cpu == 1, pos_w, neg_w)
        return WeightedRandomSampler(
            weights=weights.tolist(), num_samples=total_samples, replacement=True
        )

    def _get_val_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader for validation using train edges for message passing."""
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_attr = data.edge_attr[data.train_mask]
        val_data = Data(x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)

        return LinkNeighborLoader(
            val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=data.edge_index[:, data.val_mask],
            edge_label=data.y[data.val_mask],
            batch_size=self.batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0,
        )

    def _get_test_loader(self, data: Data) -> LinkNeighborLoader:
        """Creates a loader for testing using train+val edges for message passing."""
        history_mask = data.train_mask | data.val_mask
        test_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, history_mask],
            edge_attr=data.edge_attr[history_mask],
        )

        return LinkNeighborLoader(
            test_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=data.edge_index[:, data.test_mask],
            edge_label=data.y[data.test_mask],
            batch_size=self.batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0,
        )

    def _prepare_loss_criterion(self, data: Data) -> nn.Module:
        """Builds Weighted BCE loss, computing pos_weight from data if not provided."""
        weight_value = self.pos_weight or compute_pos_weight(data)
        self.logger.info("Using pos_weight=%.2f for BCE loss", weight_value)
        return build_weighted_bce_loss(weight_value, self.device)

    def _inject_ego_ids(self, batch: Data) -> torch.Tensor:
        """Concatenates ego-ID flags to node features for seed edge endpoints."""
        num_nodes = batch.x.size(0)
        ego_flag = torch.zeros((num_nodes, 1), device=self.device, dtype=batch.x.dtype)
        ego_flag[batch.edge_label_index[0]] = 1.0
        ego_flag[batch.edge_label_index[1]] = 1.0
        return torch.cat([batch.x, ego_flag], dim=-1)

    def _train_epoch(
        self,
        loader: LinkNeighborLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_edge_attr: torch.Tensor,
    ) -> float:
        """Trains GNN for one epoch with gradient clipping."""
        total_loss = 0.0
        total_edges = 0
        all_params = list(self.encoder.parameters()) + list(self.classifier.parameters())

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            batch_x = self._inject_ego_ids(batch)
            z = self.encoder(batch_x, batch.edge_index, batch.edge_attr)
            seed_edge_attr = train_edge_attr[batch.input_id.cpu()].to(self.device)
            out = self.classifier(z, batch.edge_label_index, seed_edge_attr)

            loss = criterion(out, batch.edge_label.float())
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=MAX_GRAD_NORM)
            optimizer.step()

            num_batch_edges = batch.edge_label_index.size(1)
            total_loss += loss.item() * num_batch_edges
            total_edges += num_batch_edges
        return total_loss / total_edges if total_edges > 0 else 0.0

    def train(self, data: Data) -> None:
        """Trains the GNN with LR scheduling and checkpoints the best model."""
        all_params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        optimizer = optim.Adam(all_params, lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=SCHEDULER_MIN_LR
        )
        criterion = self._prepare_loss_criterion(data)
        loader = self._get_train_loader(data)
        train_edge_attr = data.edge_attr[data.train_mask]
        best_pr_auc, best_enc_state, best_cls_state = -1.0, None, None

        for epoch in range(self.epochs):
            self.encoder.train()
            self.classifier.train()
            avg_loss = self._train_epoch(loader, optimizer, criterion, train_edge_attr)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.info(
                "Epoch %d/%d - Loss: %.4f - LR: %.6f",
                epoch + 1,
                self.epochs,
                avg_loss,
                current_lr,
            )

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

        self._restore_best_weights(best_enc_state, best_cls_state, best_pr_auc)

    def _restore_best_weights(
        self,
        enc_state: dict[str, torch.Tensor] | None,
        cls_state: dict[str, torch.Tensor] | None,
        best_pr_auc: float,
    ) -> None:
        """Restores the best model checkpoint from training."""
        if enc_state is not None and cls_state is not None:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in enc_state.items()})
            self.classifier.load_state_dict({k: v.to(self.device) for k, v in cls_state.items()})
            self.logger.info("Restored best model weights with PR-AUC: %.4f", best_pr_auc)

    def predict(self, data: Data, stage: str = "val") -> np.ndarray:
        """Generates sigmoid probabilities for edges at the given stage."""
        self.encoder.eval()
        self.classifier.eval()

        loader, edge_attr = self._get_loader_and_attrs_for_stage(data, stage)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                batch_x = self._inject_ego_ids(batch)
                z = self.encoder(batch_x, batch.edge_index, batch.edge_attr)
                seed_edge_attr = edge_attr[batch.input_id.cpu()].to(self.device)
                out = self.classifier(z, batch.edge_label_index, seed_edge_attr)
                preds.append(torch.sigmoid(out).cpu().numpy())
        return np.concatenate(preds)

    def _get_loader_and_attrs_for_stage(
        self, data: Data, stage: str
    ) -> tuple[LinkNeighborLoader, torch.Tensor]:
        """Returns the appropriate loader and edge attributes for a given stage."""
        stage_config: dict[str, tuple[LinkNeighborLoader, torch.Tensor]] = {
            "train": (self._get_train_loader(data), data.edge_attr[data.train_mask]),
            "val": (self._get_val_loader(data), data.edge_attr[data.val_mask]),
            "test": (self._get_test_loader(data), data.edge_attr[data.test_mask]),
        }
        if stage not in stage_config:
            raise ValueError(f"Unknown stage: {stage}")
        return stage_config[stage]

    def evaluate(self, data: Data, stage: str = "val") -> dict[str, float]:
        """Evaluates GNN predictions, optimizing the decision threshold on validation."""
        probs = self.predict(data, stage=stage)
        y_true = self._get_labels_for_stage(data, stage)

        if stage == "val":
            metrics = evaluate_predictions(probs, y_true)
            self.threshold = metrics["optimal_threshold"]
        else:
            metrics = self._compute_metrics_at_threshold(probs, y_true)

        self.logger.info(
            "%s Stage - Threshold: %.4f, F1: %.4f, PR-AUC: %.4f, ROC-AUC: %.4f",
            stage.capitalize(),
            self.threshold,
            metrics["f1_score"],
            metrics["pr_auc"],
            metrics["roc_auc"],
        )
        return metrics

    def _get_labels_for_stage(self, data: Data, stage: str) -> np.ndarray:
        """Extracts ground truth labels for the given stage mask."""
        mask_map = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
        if stage not in mask_map:
            raise ValueError(f"Unknown stage: {stage}")
        return data.y[mask_map[stage]].cpu().numpy()

    def _compute_metrics_at_threshold(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> dict[str, float]:
        """Computes classification metrics using the stored optimal threshold."""
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        preds = (probs > self.threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "optimal_threshold": self.threshold,
        }

    def get_underlying_model(self) -> object:
        """Returns the encoder and classifier tuple for serialization."""
        return (self.encoder, self.classifier)
