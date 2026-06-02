"""Weighted BCE Loss for binary classification with class imbalance."""

import torch
from torch import nn
from torch_geometric.data import Data


def compute_pos_weight(graph_data: Data) -> float:
    """Computes positive class weight from the training label distribution."""
    train_labels = graph_data.y[graph_data.train_mask]
    num_positive = (train_labels == 1).sum().float()
    num_negative = (train_labels == 0).sum().float()
    return float(num_negative / num_positive.clamp(min=1))


def build_weighted_bce_loss(pos_weight_value: float, device: torch.device) -> nn.BCEWithLogitsLoss:
    """Builds a BCEWithLogitsLoss with the specified positive class weight."""
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


def prepare_loss_criterion(
    pos_weight: float | None, graph_data: Data, device: torch.device
) -> nn.Module:
    """Builds Weighted BCE loss, computing pos_weight from data if not provided."""
    weight_value = pos_weight or compute_pos_weight(graph_data)
    return build_weighted_bce_loss(weight_value, device)
