"""Definition of MEGA-PNA graph neural network (GNN) layers and architectures."""

import typing

import torch
import torch.nn.functional as functional_interface
from torch import nn
from torch_geometric.nn import PNAConv
from torch_scatter import scatter

from src.models.gnn.config import GNNModelConfig


class MEGAPNAEncoder(nn.Module):
    """Node encoder based on MEGA-PNA.

    Implements two-stage aggregation (Multi-Edge and Neighborhood),
    reverse message passing (reverse_mp), and edge updates (emlps).
    """

    def __init__(
        self,
        config: GNNModelConfig,
        deg: torch.Tensor,
    ) -> None:
        """Initializes the MEGA-PNA layers."""
        super().__init__()
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.processed_edge_dim = (config.edge_feat_dim * 4) + 1
        self.node_proj = nn.Linear(config.in_channels, config.hidden_channels)
        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self._build_layers(config, deg)

    def _build_layers(self, config: GNNModelConfig, deg: torch.Tensor) -> None:
        """Builds convolutional layers and edge MLPs."""
        current_in_channels = config.hidden_channels
        current_edge_dim = self.processed_edge_dim
        for _ in range(config.num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=current_in_channels,
                    out_channels=config.hidden_channels,
                    aggregators=["mean", "min", "max", "std"],
                    scalers=["identity", "amplification", "attenuation"],
                    deg=deg,
                    edge_dim=current_edge_dim,
                )
            )
            mlp_in_dim = current_edge_dim + (config.hidden_channels * 2)
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(mlp_in_dim, config.hidden_channels),
                    nn.ReLU(),
                    nn.Linear(config.hidden_channels, config.hidden_channels),
                )
            )
            current_in_channels = config.hidden_channels
            current_edge_dim = config.hidden_channels

    def _flatten_edges(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-Edge Aggregation."""
        edge_idx_1d = edge_index[0] * num_nodes + edge_index[1]
        unique_edge_idx_1d, inverse_indices = torch.unique(edge_idx_1d, return_inverse=True)

        mean_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="mean")
        max_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="max")
        min_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="min")

        # Variance and std
        mean_sq = scatter(edge_attr**2, inverse_indices, dim=0, reduce="mean")
        var_attr = mean_sq - (mean_attr**2)
        std_attr = torch.sqrt(torch.clamp(var_attr, min=1e-6))

        flat_edge_attr = torch.cat([mean_attr, max_attr, min_attr, std_attr], dim=-1)

        src = unique_edge_idx_1d // num_nodes
        dst = unique_edge_idx_1d % num_nodes
        flat_edge_index = torch.stack([src, dst], dim=0)

        return flat_edge_index, flat_edge_attr

    def _reverse_mp(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Injects reverse edges for bidirectional message passing."""
        num_edges: int = edge_index.size(1)

        fwd_flags: torch.Tensor = edge_attr.new_ones((num_edges, 1))
        rev_flags: torch.Tensor = edge_attr.new_zeros((num_edges, 1))

        rev_edge_index: torch.Tensor = edge_index[[1, 0]]

        final_edge_index: torch.Tensor = torch.cat([edge_index, rev_edge_index], dim=1)

        fwd_attr: torch.Tensor = torch.cat([edge_attr, fwd_flags], dim=-1)
        rev_attr: torch.Tensor = torch.cat([edge_attr, rev_flags], dim=-1)
        final_edge_attr: torch.Tensor = torch.cat([fwd_attr, rev_attr], dim=0)

        return final_edge_index, final_edge_attr

    def _process_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply neighborhood aggregation and edge updates for a single layer."""
        x_new: torch.Tensor = self.convs[layer_idx](x, edge_index, edge_attr)
        x_new = functional_interface.relu(x_new) + x
        x_out: torch.Tensor = functional_interface.dropout(
            x_new, p=self.dropout, training=self.training
        )

        x_src: torch.Tensor = x_out[edge_index[0]]
        x_dst: torch.Tensor = x_out[edge_index[1]]
        mlp_in: torch.Tensor = torch.cat([edge_attr, x_src, x_dst], dim=-1)
        edge_attr_out: torch.Tensor = functional_interface.relu(self.edge_mlps[layer_idx](mlp_in))

        return x_out, edge_attr_out

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Perform the forward pass through the MEGA-PNA layers."""
        num_nodes: int = x.size(0)

        flat_edges: tuple[torch.Tensor, torch.Tensor] = self._flatten_edges(
            edge_index, edge_attr, num_nodes
        )
        flat_index: torch.Tensor = flat_edges[0]
        flat_attr: torch.Tensor = flat_edges[1]

        processed_edges: tuple[torch.Tensor, torch.Tensor] = self._reverse_mp(flat_index, flat_attr)
        curr_index: torch.Tensor = processed_edges[0]
        curr_attr: torch.Tensor = processed_edges[1]

        x = functional_interface.relu(self.node_proj(x))

        for i in range(self.num_layers):
            x, curr_attr = self._process_layer(i, x, curr_index, curr_attr)

        return typing.cast(torch.Tensor, x)


class EdgeClassifier(nn.Module):
    """Edge (transaction) classifier using node embeddings and edge attributes."""

    def __init__(
        self, node_emb_dim: int, edge_attr_dim: int, hidden_dim: int, final_dropout: float = 0.1
    ) -> None:
        """Initializes the classification MLP."""
        super().__init__()
        in_dim = (node_emb_dim * 2) + edge_attr_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=final_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=final_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, z: torch.Tensor, edge_label_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the probability (logits) of fraud for the given edges."""
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]

        z_src = z[src_nodes]
        z_dst = z[dst_nodes]

        edge_feat = torch.cat([z_src, z_dst, edge_attr], dim=-1)

        out = self.mlp(edge_feat).squeeze(-1)
        return typing.cast(torch.Tensor, out)
