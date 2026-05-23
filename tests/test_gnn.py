"""Tests unitarios para la arquitectura GNN y la construcción de grafos."""

import polars as pl
import pytest
import torch
from torch_geometric.data import Data  # type: ignore

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

    import os

    dataset_dir = os.path.join(tmp_path, "mock_dataset")
    os.makedirs(dataset_dir)

    accounts.write_csv(os.path.join(dataset_dir, "Mock_accounts.csv"))
    transactions.write_csv(os.path.join(dataset_dir, "Mock_Trans.csv"))

    builder = AMLGraphBuilder()
    data = builder.build_graph(dataset_dir, "Mock")

    assert isinstance(data, Data)
    assert data.num_nodes == 3
    assert data.num_edges == 2
    assert data.edge_index.shape == (2, 2)
    assert data.y.tolist() == [0, 1]


def test_gnn_forward_pass() -> None:
    """Prueba que el modelo GNN realiza una pasada hacia adelante sin errores de dimensiones."""
    num_nodes = 5
    node_feat_dim = 4
    edge_feat_dim = 3

    # Grafo sintético
    x = torch.rand((num_nodes, node_feat_dim))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.rand((4, edge_feat_dim))
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    model = GNNFraudDetector(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=8,
        num_layers=2,
        batch_size=2,
        epochs=1,
    )

    # Forzar ejecución en CPU para el test local
    model.device = torch.device("cpu")
    model.encoder.to("cpu")
    model.classifier.to("cpu")

    # Evaluar que el pipeline entrena y predice sin crashes
    model.train(data)
    preds = model.predict(data)

    assert preds.shape == (4,)
    assert all(0 <= p <= 1 for p in preds)
