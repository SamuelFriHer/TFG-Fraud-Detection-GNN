"""Full GNN model definition for fraud detection, training and evaluation."""

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Data

from src.models.gnn.data_loader import (
    compute_degree_histogram,
    get_loader_and_attrs_for_stage,
    get_train_loader,
)
from src.models.gnn.evaluator import (
    evaluate_predictions,
    evaluate_predictions_at_threshold,
    get_labels_for_stage,
)
from src.models.gnn.layers import EdgeClassifier, MEGAPNAEncoder
from src.models.gnn.loss import prepare_loss_criterion
from src.models.gnn.utils import GNNTrainingContext, predict_gnn, train_gnn_epoch
from src.models.interfaces import IGraphModel
from src.utils.logger import ProjectLogger

SCHEDULER_MIN_LR = 1e-6


class GNNFraudDetector(IGraphModel):
    """GNN model for edge-level transaction classification using MEGA-PNA."""

    def __init__(
        self,
        graph_data: Data,
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
        deg = compute_degree_histogram(graph_data)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.threshold = 0.5
        self.num_neighbors = num_neighbors or [20, 10]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_network(
            node_feat_dim,
            edge_feat_dim,
            hidden_channels,
            num_layers,
            deg,
            dropout,
            final_dropout,
        )
        self.logger = ProjectLogger.get_logger("GNNFraudDetector")

    def _init_network(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int,
        num_layers: int,
        deg: torch.Tensor,
        dropout: float,
        final_dropout: float,
    ) -> None:
        """Initializes encoder and classifier modules and transfers to device."""
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
        self.encoder.to(self.device)
        self.classifier.to(self.device)

    def _checkpoint_best(
        self,
        epoch: int,
        pr_auc: float,
        best_pr_auc: float,
        best_enc_state: dict[str, torch.Tensor] | None,
        best_cls_state: dict[str, torch.Tensor] | None,
    ) -> tuple[float, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """Checkpoints GNN model weights if the validation PR-AUC improves."""
        if pr_auc > best_pr_auc:
            best_enc_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
            best_cls_state = {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()}
            self.logger.info("Best model updated at epoch %d (Val PR-AUC: %.4f)", epoch + 1, pr_auc)
            return pr_auc, best_enc_state, best_cls_state
        return best_pr_auc, best_enc_state, best_cls_state

    def train(self, graph_data: Data) -> None:
        """Trains the GNN with LR scheduling and checkpoints the best model."""
        all_params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        optimizer = optim.Adam(all_params, lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=SCHEDULER_MIN_LR
        )
        criterion = prepare_loss_criterion(self.pos_weight, graph_data, self.device)
        loader = get_train_loader(graph_data, self.num_neighbors, self.batch_size)
        train_edge_attr = graph_data.edge_attr[graph_data.train_mask]
        best_pr_auc, best_enc_state, best_cls_state = -1.0, None, None
        context = GNNTrainingContext(
            self.encoder,
            self.classifier,
            loader,
            optimizer,
            criterion,
            train_edge_attr,
            self.device,
        )

        for epoch in range(self.epochs):
            avg_loss = train_gnn_epoch(context)
            scheduler.step()
            self.logger.info(
                "Epoch %d/%d - Loss: %.4f - LR: %.6f",
                epoch + 1,
                self.epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )
            metrics = self.evaluate(graph_data, stage="val")
            best_pr_auc, best_enc_state, best_cls_state = self._checkpoint_best(
                epoch, metrics["pr_auc"], best_pr_auc, best_enc_state, best_cls_state
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

    def predict(self, graph_data: Data, stage: str = "val") -> np.ndarray:
        """Generates sigmoid probabilities for edges at the given stage."""
        loader, edge_attr = get_loader_and_attrs_for_stage(
            graph_data, stage, self.num_neighbors, self.batch_size
        )
        return predict_gnn(self.encoder, self.classifier, loader, edge_attr, self.device)

    def evaluate(self, graph_data: Data, stage: str = "val") -> dict[str, float]:
        """Evaluates GNN predictions, optimizing the decision threshold on validation."""
        probs = self.predict(graph_data, stage=stage)
        y_true = get_labels_for_stage(graph_data, stage)

        if stage == "val":
            metrics = evaluate_predictions(probs, y_true)
            self.threshold = metrics["optimal_threshold"]
        else:
            metrics = evaluate_predictions_at_threshold(probs, y_true, self.threshold)

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
