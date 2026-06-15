"""Unit tests for the GNN loss function builders."""

import torch
from torch import nn
from torch_geometric.data import Data

from src.models.gnn.loss import (
    build_weighted_bce_loss,
    compute_pos_weight,
    prepare_loss_criterion,
)


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


def test_compute_pos_weight() -> None:
    """Verify that compute_pos_weight correctly calculates the positive class weight."""
    # Case 1: Balanced labels in training set (4 zeros, 4 ones)
    labels_balanced: torch.Tensor = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    mask_all: torch.Tensor = torch.tensor([True] * 8, dtype=torch.bool)
    data_balanced: Data = Data(y=labels_balanced, train_mask=mask_all)
    weight_balanced: float = compute_pos_weight(data_balanced)
    assert weight_balanced == 1.0

    # Case 2: Imbalanced labels (8 zeros, 2 ones)
    labels_imbalanced: torch.Tensor = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long)
    mask_imbalanced: torch.Tensor = torch.tensor([True] * 10, dtype=torch.bool)
    data_imbalanced: Data = Data(y=labels_imbalanced, train_mask=mask_imbalanced)
    weight_imbalanced: float = compute_pos_weight(data_imbalanced)
    assert weight_imbalanced == 4.0

    # Case 3: Mask respects subset (only indices with True are used)
    labels_masked: torch.Tensor = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    mask_subset: torch.Tensor = torch.tensor(
        [True, True, True, False, False, False], dtype=torch.bool
    )
    data_masked: Data = Data(y=labels_masked, train_mask=mask_subset)
    weight_masked: float = compute_pos_weight(data_masked)
    assert weight_masked == 2.0

    # Case 4: No positive labels in train set (clamping prevents division by zero)
    labels_no_pos: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.long)
    mask_no_pos: torch.Tensor = torch.tensor([True, True, True], dtype=torch.bool)
    data_no_pos: Data = Data(y=labels_no_pos, train_mask=mask_no_pos)
    weight_no_pos: float = compute_pos_weight(data_no_pos)
    assert weight_no_pos == 3.0


def test_prepare_loss_criterion() -> None:
    """Verify that prepare_loss_criterion uses provided weight or computes it."""
    labels: torch.Tensor = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    mask: torch.Tensor = torch.tensor([True] * 4, dtype=torch.bool)
    graph_data: Data = Data(y=labels, train_mask=mask)
    device: torch.device = torch.device("cpu")

    # If pos_weight is provided, use it
    loss_fn_provided: nn.Module = prepare_loss_criterion(5.0, graph_data, device)
    assert isinstance(loss_fn_provided, nn.BCEWithLogitsLoss)
    assert loss_fn_provided.pos_weight is not None
    assert torch.allclose(loss_fn_provided.pos_weight, torch.tensor([5.0], device=device))

    # If pos_weight is None, compute from data
    loss_fn_computed: nn.Module = prepare_loss_criterion(None, graph_data, device)
    assert isinstance(loss_fn_computed, nn.BCEWithLogitsLoss)
    assert loss_fn_computed.pos_weight is not None
    assert torch.allclose(loss_fn_computed.pos_weight, torch.tensor([3.0], device=device))
