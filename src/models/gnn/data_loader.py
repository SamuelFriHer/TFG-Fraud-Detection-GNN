"""Data loading and batch construction utilities for the GNN model."""

import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import degree


def compute_degree_histogram(graph_data: Data) -> torch.Tensor:
    """Computes in-degree histogram from training edges for PNAConv."""
    train_edge_index = graph_data.edge_index[:, graph_data.train_mask]
    in_degree = degree(train_edge_index[1], graph_data.num_nodes, dtype=torch.long)
    return torch.bincount(in_degree)


def build_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler | None:
    """Builds a WeightedRandomSampler for class-balanced mini-batches."""
    y_cpu = labels.cpu()
    num_pos = int((y_cpu == 1).sum().item())
    num_neg = int((y_cpu == 0).sum().item())
    if num_pos == 0 or num_neg == 0:
        return None

    total_samples = num_pos + num_neg
    pos_w = total_samples / (2.0 * num_pos)
    neg_w = total_samples / (2.0 * num_neg)
    weights = torch.where(y_cpu == 1, pos_w, neg_w)
    return WeightedRandomSampler(
        weights=weights.tolist(), num_samples=total_samples, replacement=True
    )


def _create_link_loader(
    data: Data,
    num_neighbors: list[int],
    batch_size: int,
    sampler: WeightedRandomSampler | None,
) -> LinkNeighborLoader:
    """Constructs a LinkNeighborLoader using either a sampler or shuffling."""
    if sampler is not None:
        return LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=data.edge_index,
            edge_label=data.y,
            batch_size=batch_size,
            sampler=sampler,
            neg_sampling_ratio=0.0,
        )
    return LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=data.edge_index,
        edge_label=data.y,
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=0.0,
    )


def get_train_loader(
    graph_data: Data, num_neighbors: list[int], batch_size: int
) -> LinkNeighborLoader:
    """Creates a loader with weighted sampling for class-balanced training."""
    train_edge_index = graph_data.edge_index[:, graph_data.train_mask]
    train_edge_attr = graph_data.edge_attr[graph_data.train_mask]
    train_y = graph_data.y[graph_data.train_mask]

    train_data = Data(
        x=graph_data.x,
        edge_index=train_edge_index,
        edge_attr=train_edge_attr,
        y=train_y,
    )

    sampler = build_weighted_sampler(train_y)
    return _create_link_loader(train_data, num_neighbors, batch_size, sampler)


def get_val_loader(
    graph_data: Data, num_neighbors: list[int], batch_size: int
) -> LinkNeighborLoader:
    """Creates a loader for validation using train edges for message passing."""
    train_edge_index = graph_data.edge_index[:, graph_data.train_mask]
    train_edge_attr = graph_data.edge_attr[graph_data.train_mask]
    val_data = Data(x=graph_data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)

    return LinkNeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        edge_label_index=graph_data.edge_index[:, graph_data.val_mask],
        edge_label=graph_data.y[graph_data.val_mask],
        batch_size=batch_size,
        shuffle=False,
        neg_sampling_ratio=0.0,
    )


def get_test_loader(
    graph_data: Data, num_neighbors: list[int], batch_size: int
) -> LinkNeighborLoader:
    """Creates a loader for testing using train+val edges for message passing."""
    history_mask = graph_data.train_mask | graph_data.val_mask
    test_data = Data(
        x=graph_data.x,
        edge_index=graph_data.edge_index[:, history_mask],
        edge_attr=graph_data.edge_attr[history_mask],
    )

    return LinkNeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        edge_label_index=graph_data.edge_index[:, graph_data.test_mask],
        edge_label=graph_data.y[graph_data.test_mask],
        batch_size=batch_size,
        shuffle=False,
        neg_sampling_ratio=0.0,
    )


def get_loader_and_attrs_for_stage(
    graph_data: Data, stage: str, num_neighbors: list[int], batch_size: int
) -> tuple[LinkNeighborLoader, torch.Tensor]:
    """Returns the appropriate loader and edge attributes for a given stage."""
    stage_config: dict[str, tuple[LinkNeighborLoader, torch.Tensor]] = {
        "train": (
            get_train_loader(graph_data, num_neighbors, batch_size),
            graph_data.edge_attr[graph_data.train_mask],
        ),
        "val": (
            get_val_loader(graph_data, num_neighbors, batch_size),
            graph_data.edge_attr[graph_data.val_mask],
        ),
        "test": (
            get_test_loader(graph_data, num_neighbors, batch_size),
            graph_data.edge_attr[graph_data.test_mask],
        ),
    }
    if stage not in stage_config:
        raise ValueError(f"Unknown stage: {stage}")
    return stage_config[stage]
