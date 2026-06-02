"""GNN training, inference and feature manipulation utilities."""

import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader


def inject_ego_ids(subgraph: Data, device: torch.device) -> torch.Tensor:
    """Concatenates ego-ID flags to node features for seed edge endpoints."""
    num_nodes = subgraph.x.size(0)
    ego_flag = torch.zeros((num_nodes, 1), device=device, dtype=subgraph.x.dtype)
    ego_flag[subgraph.edge_label_index[0]] = 1.0
    ego_flag[subgraph.edge_label_index[1]] = 1.0
    return torch.cat([subgraph.x, ego_flag], dim=-1)


def train_gnn_epoch(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: LinkNeighborLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_edge_attr: torch.Tensor,
    device: torch.device,
) -> float:
    """Trains GNN encoder and classifier for one epoch with gradient clipping."""
    encoder.train()
    classifier.train()
    total_loss = 0.0
    total_edges = 0
    all_params = list(encoder.parameters()) + list(classifier.parameters())

    for subgraph in loader:
        subgraph = subgraph.to(device)
        optimizer.zero_grad()

        batch_x = inject_ego_ids(subgraph, device)
        z = encoder(batch_x, subgraph.edge_index, subgraph.edge_attr)
        seed_edge_attr = train_edge_attr[subgraph.input_id.cpu()].to(device)
        out = classifier(z, subgraph.edge_label_index, seed_edge_attr)

        loss = criterion(out, subgraph.edge_label.float())
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        num_batch_edges = subgraph.edge_label_index.size(1)
        total_loss += loss.item() * num_batch_edges
        total_edges += num_batch_edges

    return total_loss / total_edges if total_edges > 0 else 0.0


def predict_gnn(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: LinkNeighborLoader,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Generates sigmoid probabilities for edges using GNN encoder/classifier."""
    encoder.eval()
    classifier.eval()
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for subgraph in loader:
            subgraph = subgraph.to(device)
            batch_x = inject_ego_ids(subgraph, device)
            z = encoder(batch_x, subgraph.edge_index, subgraph.edge_attr)
            seed_edge_attr = edge_attr[subgraph.input_id.cpu()].to(device)
            out = classifier(z, subgraph.edge_label_index, seed_edge_attr)
            preds.append(torch.sigmoid(out).cpu().numpy())

    return np.concatenate(preds)
