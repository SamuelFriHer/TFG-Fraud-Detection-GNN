"""Definition of MEGA-PNA graph neural network (GNN) layers and architectures."""

from __future__ import annotations

import typing

import torch
import torch.nn.functional as functional_interface
from torch import nn
from torch_geometric.nn import MessagePassing, PNAConv
from torch_scatter import scatter

from src.models.gnn.config import GNNModelConfig


class MEGAPNAEncoder(nn.Module):
    """Node encoder based on MEGA-PNA GNN."""

    def __init__(
        self,
        config: GNNModelConfig,
        deg: torch.Tensor,
    ) -> None:
        """Initializes the MEGA-PNA layers."""
        super().__init__()
        self.num_layers: int = config.num_layers
        self.dropout: float = config.dropout
        self.processed_edge_dim: int = (config.edge_feat_dim * 4) + 1
        self.node_proj: nn.Linear = nn.Linear(config.in_channels, config.hidden_channels)
        self.convs: nn.ModuleList = nn.ModuleList()
        self.edge_mlps: nn.ModuleList = nn.ModuleList()
        self._build_layers(config, deg)
        self.current_inverse_indices: torch.Tensor | None = None
        self._setup_explain_hooks()

    def _build_layers(self, config: GNNModelConfig, deg: torch.Tensor) -> None:
        """Builds convolutional layers and edge MLPs."""
        current_in_channels: int = config.hidden_channels
        current_edge_dim: int = self.processed_edge_dim
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
            mlp_in_dim: int = current_edge_dim + (config.hidden_channels * 2)
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(mlp_in_dim, config.hidden_channels),
                    nn.ReLU(),
                    nn.Linear(config.hidden_channels, config.hidden_channels),
                )
            )
            current_in_channels = config.hidden_channels
            current_edge_dim = config.hidden_channels

    def _setup_explain_hooks(self) -> None:
        """Hooks custom explain_message to convolutions to handle modified edge index."""
        for conv in self.convs:
            if isinstance(conv, MessagePassing):
                conv.explain_message = self._make_custom_explain_message(conv)

    def _make_custom_explain_message(
        self, conv: MessagePassing
    ) -> typing.Callable[[torch.Tensor, int | None], torch.Tensor]:
        """Creates a custom explain_message function to handle modified edge index."""

        def custom_explain_message(
            conv_self: MessagePassing, inputs: torch.Tensor, dim_size: int | None
        ) -> torch.Tensor:
            if (edge_mask := getattr(conv_self, "_edge_mask", None)) is None:
                raise ValueError("Could not find a pre-defined 'edge_mask' to explain.")
            if getattr(conv_self, "_apply_sigmoid", True):
                edge_mask = edge_mask.sigmoid()
            if (inv_idx := self.current_inverse_indices) is None:
                raise ValueError("current_inverse_indices is not set in MEGAPNAEncoder.")
            flat_mask: torch.Tensor = scatter(edge_mask, inv_idx, dim=0, reduce="mean")
            mapped_mask: torch.Tensor = torch.cat([flat_mask, flat_mask], dim=0)
            node_dim: int = conv_self.node_dim
            if inputs.size(node_dim) != mapped_mask.size(0):
                msg = f"Dim mismatch: {inputs.size(node_dim)} vs {mapped_mask.size(0)}"
                raise AssertionError(msg)
            size: list[int] = [1] * inputs.dim()
            size[node_dim] = -1
            return inputs * mapped_mask.view(size)

        custom_explain_message.__name__ = "explain_message"
        return custom_explain_message.__get__(conv, conv.__class__)

    def _flatten_edges(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-Edge Aggregation."""
        edge_idx_1d: torch.Tensor = edge_index[0] * num_nodes + edge_index[1]
        unique_res: tuple[torch.Tensor, torch.Tensor] = torch.unique(
            edge_idx_1d, return_inverse=True
        )
        unique_edge_idx_1d: torch.Tensor = unique_res[0]
        inverse_indices: torch.Tensor = unique_res[1]

        mean: torch.Tensor = scatter(edge_attr, inverse_indices, dim=0, reduce="mean")
        std: torch.Tensor = torch.sqrt(
            torch.clamp(
                scatter(edge_attr**2, inverse_indices, dim=0, reduce="mean") - mean**2,
                min=1e-6,
            )
        )
        flat_edge_attr: torch.Tensor = torch.cat(
            [
                mean,
                scatter(edge_attr, inverse_indices, dim=0, reduce="max"),
                scatter(edge_attr, inverse_indices, dim=0, reduce="min"),
                std,
            ],
            dim=-1,
        )

        src: torch.Tensor = unique_edge_idx_1d // num_nodes
        dst: torch.Tensor = unique_edge_idx_1d % num_nodes
        return torch.stack([src, dst], dim=0), flat_edge_attr, inverse_indices

    def _reverse_mp(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Injects reverse edges for bidirectional message passing."""
        num_edges: int = edge_index.size(1)
        fwd_attr: torch.Tensor = torch.cat([edge_attr, edge_attr.new_ones((num_edges, 1))], dim=-1)
        rev_attr: torch.Tensor = torch.cat([edge_attr, edge_attr.new_zeros((num_edges, 1))], dim=-1)
        return (
            torch.cat([edge_index, edge_index[[1, 0]]], dim=1),
            torch.cat([fwd_attr, rev_attr], dim=0),
        )

    def _process_layer(
        self, layer_idx: int, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply neighborhood aggregation and edge updates for a single layer."""
        x_new: torch.Tensor = (
            functional_interface.relu(self.convs[layer_idx](x, edge_index, edge_attr)) + x
        )
        x_out: torch.Tensor = functional_interface.dropout(
            x_new, p=self.dropout, training=self.training
        )
        mlp_in: torch.Tensor = torch.cat(
            [edge_attr, x_out[edge_index[0]], x_out[edge_index[1]]], dim=-1
        )
        return x_out, functional_interface.relu(self.edge_mlps[layer_idx](mlp_in))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Perform the forward pass through the MEGA-PNA layers."""
        num_nodes: int = x.size(0)

        flat_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self._flatten_edges(
            edge_index, edge_attr, num_nodes
        )
        flat_index: torch.Tensor = flat_edges[0]
        flat_attr: torch.Tensor = flat_edges[1]
        inverse_indices: torch.Tensor = flat_edges[2]

        processed_edges: tuple[torch.Tensor, torch.Tensor] = self._reverse_mp(flat_index, flat_attr)
        curr_index: torch.Tensor = processed_edges[0]
        curr_attr: torch.Tensor = processed_edges[1]

        self.current_inverse_indices = inverse_indices

        h: torch.Tensor = functional_interface.relu(self.node_proj(x))

        for i in range(self.num_layers):
            h, curr_attr = self._process_layer(i, h, curr_index, curr_attr)

        return typing.cast(torch.Tensor, h)
