"""Focal Loss implementation for binary classification."""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss module for binary classification to address class imbalance.

    Focuses model training on hard/misclassified samples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initializes the FocalLoss instance.

        Args:
            alpha: Weighting factor for the positive class.
            gamma: Focusing parameter for hard examples.
            reduction: Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss between inputs (logits) and targets.

        Args:
            inputs: Logits predicted by the model.
            targets: Binary labels.

        Returns:
            The computed Focal Loss.
        """
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
