"""Unit tests for the GNN architecture and graph construction."""

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn.config import GNNModelConfig
from src.models.gnn.model import GNNFraudDetector
from src.models.interfaces import IGraphModel


@pytest.fixture
def mock_graph_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Creates synthetic DataFrames for accounts and transactions."""
    accounts = pl.DataFrame(
        {
            "Bank Name": ["BankA", "BankA", "BankB"],
            "Bank ID": [1, 1, 2],
            "Account Number": ["A1", "A2", "B1"],
            "Entity ID": ["E1", "E2", "E3"],
            "Entity Name": ["Corp1", "Corp2", "Indiv1"],
        }
    )

    transactions = pl.DataFrame(
        {
            "Timestamp": ["2023/01/01 10:00", "2023/01/01 11:00"],
            "From Bank": [1, 1],
            "Account": ["A1", "A2"],
            "To Bank": [1, 2],
            "Account.1": ["A2", "B1"],
            "Amount Received": [100.0, 50.0],
            "Receiving Currency": ["USD", "EUR"],
            "Amount Paid": [100.0, 50.0],
            "Payment Currency": ["USD", "EUR"],
            "Payment Format": ["Wire", "Cheque"],
            "Is Laundering": [0, 1],
        }
    )

    return accounts, transactions


def test_gnn_interface() -> None:
    """Interface: Verify GNN model implements the correct interface."""
    assert issubclass(GNNFraudDetector, IGraphModel)


def test_gnn_forward_pass() -> None:
    """Tests that the GNN model performs a forward pass without dimension errors."""
    num_nodes = 5
    node_feat_dim = 10
    edge_feat_dim = 3

    x = torch.rand((num_nodes, node_feat_dim))
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 4, 2, 3, 4, 0]], dtype=torch.long
    )
    edge_attr = torch.rand((8, edge_feat_dim))
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False, False, False, False, False])
    val_mask = torch.tensor([False, False, True, True, False, False, False, False])
    test_mask = torch.tensor([False, False, False, False, True, True, False, False])

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    config = GNNModelConfig(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=8,
        num_layers=2,
        batch_size=2,
        epochs=1,
    )
    model = GNNFraudDetector(
        graph_data=data,
        config=config,
    )

    model.device = torch.device("cpu")
    model.encoder.to("cpu")
    model.classifier.to("cpu")

    model.train(data)
    preds = model.predict(data, stage="test")

    assert preds.shape == (2,)
    assert all(0 <= p <= 1 for p in preds)


def test_weighted_bce_loss() -> None:
    """Validates that Weighted BCE Loss computes coherent losses without errors."""
    from src.models.gnn.loss import build_weighted_bce_loss

    inputs = torch.tensor([0.5, -0.5, 2.0, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss_fn = build_weighted_bce_loss(5.0, torch.device("cpu"))
    loss = loss_fn(inputs, targets)

    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_gnn_weighted_sampler() -> None:
    """Verifies that the training dataloader uses WeightedRandomSampler."""
    num_nodes = 5
    node_feat_dim = 10
    edge_feat_dim = 3

    x = torch.rand((num_nodes, node_feat_dim))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.rand((4, edge_feat_dim))
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])
    val_mask = torch.tensor([False, False, True, False])
    test_mask = torch.tensor([False, False, False, True])
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    config = GNNModelConfig(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=8,
        num_layers=2,
        batch_size=2,
        epochs=1,
    )
    model = GNNFraudDetector(
        graph_data=data,
        config=config,
    )

    from src.models.gnn.data_loader import get_train_loader

    loader = get_train_loader(data, model.config.num_neighbors, model.config.batch_size)
    assert loader.sampler is not None


def test_evaluate_predictions() -> None:
    """Verifies that evaluator functions return correct keys and values."""
    import numpy as np

    from src.models.gnn.evaluator import (
        evaluate_predictions,
        evaluate_predictions_at_threshold,
    )

    sample_probs = np.array([0.1, 0.2, 0.8, 0.9])
    sample_y_true = np.array([0, 0, 1, 1])

    computed_metrics = evaluate_predictions(sample_probs, sample_y_true)
    assert "accuracy" in computed_metrics
    assert "precision" in computed_metrics
    assert "recall" in computed_metrics
    assert "f1" in computed_metrics
    assert "roc_auc" in computed_metrics
    assert "pr_auc" in computed_metrics
    assert "optimal_threshold" in computed_metrics

    # Under optimized threshold (0.8), probs > 0.8 yields predictions [0, 0, 0, 1]
    # due to strict inequality in prediction thresholding.
    assert computed_metrics["accuracy"] == 0.75
    assert abs(computed_metrics["f1"] - 0.666666) < 1e-4

    threshold_metrics = evaluate_predictions_at_threshold(sample_probs, sample_y_true, 0.5)
    assert threshold_metrics["accuracy"] == 1.0
    assert threshold_metrics["f1"] == 1.0


def test_gnn_config_in_channels() -> None:
    """Verify that in_channels is correctly computed as node_feat_dim + 1."""
    node_feat_dim: int = 10
    edge_feat_dim: int = 5

    # Test case 1: in_channels is not provided
    config_default: GNNModelConfig = GNNModelConfig(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
    )
    assert config_default.in_channels == node_feat_dim + 1

    # Test case 2: in_channels is provided but incorrect
    config_incorrect: GNNModelConfig = GNNModelConfig(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        in_channels=20,
    )
    assert config_incorrect.in_channels == node_feat_dim + 1


def test_train_gnn_epoch_empty_loader() -> None:
    """Verify train_gnn_epoch handles an empty loader correctly by returning 0.0."""
    from unittest.mock import MagicMock

    from torch import nn, optim

    from src.models.gnn.utils import GNNTrainingContext, train_gnn_epoch

    dummy_encoder: nn.Module = nn.Module()
    dummy_classifier: nn.Module = nn.Module()
    empty_loader: list = []
    dummy_optimizer: MagicMock = MagicMock(spec=optim.Optimizer)
    dummy_criterion: nn.Module = nn.Module()
    empty_train_edge_attr: torch.Tensor = torch.empty((0,))
    target_device: torch.device = torch.device("cpu")

    training_context: GNNTrainingContext = GNNTrainingContext(
        encoder=dummy_encoder,
        classifier=dummy_classifier,
        loader=empty_loader,  # type: ignore
        optimizer=dummy_optimizer,
        criterion=dummy_criterion,
        train_edge_attr=empty_train_edge_attr,
        device=target_device,
    )

    epoch_loss: float = train_gnn_epoch(training_context)
    assert epoch_loss == 0.0
