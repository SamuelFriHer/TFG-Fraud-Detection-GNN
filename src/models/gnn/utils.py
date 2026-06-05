from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader


@dataclass(frozen=True)
class GNNTrainingContext:
    """Encapsulates context parameters required for GNN training epochs."""

    encoder: nn.Module
    classifier: nn.Module
    loader: LinkNeighborLoader
    optimizer: optim.Optimizer
    criterion: nn.Module
    train_edge_attr: torch.Tensor
    device: torch.device


def inject_ego_ids(subgraph: Data, device: torch.device) -> torch.Tensor:
    """Concatenates ego-ID flags to node features for seed edge endpoints."""
    num_nodes = subgraph.x.size(0)
    ego_flag = torch.zeros((num_nodes, 1), device=device, dtype=subgraph.x.dtype)
    ego_flag[subgraph.edge_label_index[0]] = 1.0
    ego_flag[subgraph.edge_label_index[1]] = 1.0
    return torch.cat([subgraph.x, ego_flag], dim=-1)


def train_gnn_epoch(context: GNNTrainingContext) -> float:
    """Trains GNN encoder and classifier for one epoch with gradient clipping."""
    context.encoder.train()
    context.classifier.train()
    total_loss = 0.0
    total_edges = 0
    all_params = list(context.encoder.parameters()) + list(context.classifier.parameters())

    for subgraph in context.loader:
        subgraph = subgraph.to(context.device)
        context.optimizer.zero_grad()

        batch_x = inject_ego_ids(subgraph, context.device)
        z = context.encoder(batch_x, subgraph.edge_index, subgraph.edge_attr)
        seed_edge_attr = context.train_edge_attr[subgraph.input_id.cpu()].to(context.device)
        out = context.classifier(z, subgraph.edge_label_index, seed_edge_attr)

        loss = context.criterion(out, subgraph.edge_label.float())
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        context.optimizer.step()

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
