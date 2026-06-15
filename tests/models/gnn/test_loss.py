"""Unit tests for the GNN loss function builders."""

import torch
from torch import nn

from src.models.gnn.loss import build_weighted_bce_loss


def test_build_weighted_bce_loss_factory() -> None:
    """Verify that build_weighted_bce_loss returns BCEWithLogitsLoss with correct pos_weight."""
    pos_weight: float = 3.5
    devices: list[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        loss_fn: nn.BCEWithLogitsLoss = build_weighted_bce_loss(pos_weight, device)

        # Check return class
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

        # Check pos_weight existence and device placement
        assert loss_fn.pos_weight is not None
        assert loss_fn.pos_weight.device.type == device.type

        # Check weight value correctness
        expected_weight: torch.Tensor = torch.tensor([pos_weight], device=device)
        assert torch.allclose(loss_fn.pos_weight, expected_weight)
