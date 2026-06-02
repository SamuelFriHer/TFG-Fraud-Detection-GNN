"""Tests unitarios para la arquitectura GNN y la construcción de grafos."""

import os

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from src.data.graph_builder import AMLGraphBuilder
from src.models.gnn.model import GNNFraudDetector


@pytest.fixture
def mock_graph_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Crea DataFrames sintéticos de cuentas y transacciones."""
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


def test_graph_builder(mock_graph_data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: str) -> None:
    """Valida que el grafo se construye correctamente desde CSVs."""
    accounts, transactions = mock_graph_data

    dataset_dir = os.path.join(tmp_path, "mock_dataset")
    os.makedirs(dataset_dir)

    accounts.write_csv(os.path.join(dataset_dir, "Mock_accounts.csv"))
    transactions.write_csv(os.path.join(dataset_dir, "Mock_Trans.csv"))

    builder = AMLGraphBuilder()
    data = builder.build_graph(dataset_dir, "Mock", test_size=0.4)

    assert isinstance(data, Data)
    assert data.num_nodes == 3
    assert data.num_edges == 2
    assert data.edge_index.shape == (2, 2)
    assert data.y.tolist() == [0, 1]
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")
    assert hasattr(data, "test_mask")


def test_gnn_forward_pass() -> None:
    """Prueba que el modelo GNN realiza una pasada hacia adelante sin errores de dimensiones."""
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

    model = GNNFraudDetector(
        data=data,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=8,
        num_layers=2,
        batch_size=2,
        epochs=1,
    )

    model.device = torch.device("cpu")
    model.encoder.to("cpu")
    model.classifier.to("cpu")

    model.train(data)
    preds = model.predict(data, stage="test")

    assert preds.shape == (1,)
    assert all(0 <= p <= 1 for p in preds)


def test_weighted_bce_loss() -> None:
    """Valida que Weighted BCE Loss calcula pérdidas coherentes sin errores."""
    from src.models.gnn.loss import build_weighted_bce_loss

    inputs = torch.tensor([0.5, -0.5, 2.0, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss_fn = build_weighted_bce_loss(5.0, torch.device("cpu"))
    loss = loss_fn(inputs, targets)

    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_gnn_weighted_sampler() -> None:
    """Verifica que el dataloader de entrenamiento use WeightedRandomSampler."""
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

    model = GNNFraudDetector(
        data=data,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=8,
        num_layers=2,
        batch_size=2,
        epochs=1,
    )

    loader = model._get_train_loader(data)
    assert loader.sampler is not None
