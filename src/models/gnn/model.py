"""Full GNN model definition for fraud detection, training and evaluation."""

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Data

from src.models.gnn.classifier import EdgeClassifier
from src.models.gnn.config import GNNModelConfig
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
from src.models.gnn.layers import MEGAPNAEncoder
from src.models.gnn.loss import prepare_loss_criterion
from src.models.gnn.utils import (
    GNNCheckpointState,
    GNNTrainingContext,
    predict_gnn,
    train_gnn_epoch,
)
from src.models.interfaces import IGraphModel
from src.utils.logger import ProjectLogger

SCHEDULER_MIN_LR = 1e-6


class GNNFraudDetector(IGraphModel):
    """GNN model for edge-level transaction classification using MEGA-PNA."""

    def __init__(
        self,
        graph_data: Data,
        config: GNNModelConfig,
    ) -> None:
        """Initializes MEGA-PNA encoder, edge classifier and training config."""
        deg = compute_degree_histogram(graph_data)
        self.config = config
        self.threshold = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_network(config, deg)
        self.logger = ProjectLogger.get_logger("GNNFraudDetector")

    def _init_network(
        self,
        config: GNNModelConfig,
        deg: torch.Tensor,
    ) -> None:
        """Initializes encoder and classifier modules and transfers to device."""
        self.encoder = MEGAPNAEncoder(config, deg)
        self.classifier = EdgeClassifier(
            config.hidden_channels,
            config.edge_feat_dim,
            config.hidden_channels,
            final_dropout=config.final_dropout,
        )
        self.encoder.to(self.device)
        self.classifier.to(self.device)

    def _checkpoint_best(
        self,
        epoch: int,
        pr_auc: float,
        state: GNNCheckpointState,
    ) -> GNNCheckpointState:
        """Checkpoints GNN model weights if the validation PR-AUC improves."""
        if pr_auc > state.best_pr_auc:
            best_enc = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
            best_cls = {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()}
            self.logger.info("Best model updated at epoch %d (Val PR-AUC: %.4f)", epoch + 1, pr_auc)
            return GNNCheckpointState(pr_auc, best_enc, best_cls)
        return state

    def train(self, graph_data: Data) -> None:
        """Trains the GNN with LR scheduling and checkpoints the best model."""
        all_params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        optimizer = optim.Adam(all_params, lr=self.config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=SCHEDULER_MIN_LR
        )
        criterion = prepare_loss_criterion(self.config.pos_weight, graph_data, self.device)
        context = self._create_training_context(graph_data, optimizer, criterion)
        self._run_training_loop(context, optimizer, scheduler, graph_data)

    def _create_training_context(
        self,
        graph_data: Data,
        optimizer: optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> GNNTrainingContext:
        """Creates the training context for the training epochs."""
        loader = get_train_loader(graph_data, self.config.num_neighbors, self.config.batch_size)
        train_edge_attr: torch.Tensor = graph_data.edge_attr[graph_data.train_mask].to(self.device)
        return GNNTrainingContext(
            self.encoder,
            self.classifier,
            loader,
            optimizer,
            criterion,
            train_edge_attr,
            self.device,
        )

    def _run_training_loop(
        self,
        context: GNNTrainingContext,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.CosineAnnealingLR,
        graph_data: Data,
    ) -> None:
        """Runs the epoch training loop, scheduling learning rate and saving weights."""
        state = GNNCheckpointState(-1.0, None, None)
        for epoch in range(self.config.epochs):
            avg_loss = train_gnn_epoch(context)
            scheduler.step()
            self.logger.info(
                "Epoch %d/%d - Loss: %.4f - LR: %.6f",
                epoch + 1,
                self.config.epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )
            metrics = self.evaluate(graph_data, stage="val")
            state = self._checkpoint_best(epoch, metrics["pr_auc"], state)
        self._restore_best_weights(state.best_enc_state, state.best_cls_state, state.best_pr_auc)

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
            graph_data, stage, self.config.num_neighbors, self.config.batch_size
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
            metrics["f1"],
            metrics["pr_auc"],
            metrics["roc_auc"],
        )
        return metrics

    def get_underlying_model(self) -> object:
        """Returns the encoder and classifier tuple for serialization."""
        return (self.encoder, self.classifier)
