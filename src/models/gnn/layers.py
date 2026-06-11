"""Definición de las capas y arquitecturas de red neuronal de grafos (GNN) tipo MEGA-PNA."""

import typing

import torch
import torch.nn.functional as functional_interface
from torch import nn
from torch_geometric.nn import PNAConv
from torch_scatter import scatter

from src.models.gnn.config import GNNModelConfig


class MEGAPNAEncoder(nn.Module):
    """Codificador de nodos basado en MEGA-PNA.

    Implementa agregación en dos etapas (Multi-Edge y Neighborhood),
    paso de mensajes inverso (reverse_mp) y actualización de aristas (emlps).
    """

    def __init__(
        self,
        config: GNNModelConfig,
        deg: torch.Tensor,
    ) -> None:
        """Inicializa las capas MEGA-PNA."""
        super().__init__()
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.processed_edge_dim = (config.edge_feat_dim * 4) + 1
        self.node_proj = nn.Linear(config.in_channels, config.hidden_channels)
        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self._build_layers(config, deg)

    def _build_layers(self, config: GNNModelConfig, deg: torch.Tensor) -> None:
        """Construye las capas convolucionales y MLPs de aristas."""
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
        """Agregación de aristas múltiples (Multi-Edge Aggregation)."""
        edge_idx_1d = edge_index[0] * num_nodes + edge_index[1]
        unique_edge_idx_1d, inverse_indices = torch.unique(edge_idx_1d, return_inverse=True)

        mean_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="mean")
        max_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="max")
        min_attr = scatter(edge_attr, inverse_indices, dim=0, reduce="min")

        # Varianza y std
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
        """Inyección de aristas inversas para paso de mensajes bidireccional."""
        num_edges: int = edge_index.size(1)

        fwd_flags: torch.Tensor = edge_attr.new_ones((num_edges, 1))
        rev_flags: torch.Tensor = edge_attr.new_zeros((num_edges, 1))

        rev_edge_index: torch.Tensor = edge_index[[1, 0]]

        final_edge_index: torch.Tensor = torch.cat([edge_index, rev_edge_index], dim=1)

        fwd_attr: torch.Tensor = torch.cat([edge_attr, fwd_flags], dim=-1)
        rev_attr: torch.Tensor = torch.cat([edge_attr, rev_flags], dim=-1)
        final_edge_attr: torch.Tensor = torch.cat([fwd_attr, rev_attr], dim=0)

        return final_edge_index, final_edge_attr

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Realiza el paso forward a través de las capas MEGA-PNA."""
        num_nodes = x.size(0)

        # 1. Multi-Edge Aggregation
        flat_edge_index, flat_edge_attr = self._flatten_edges(edge_index, edge_attr, num_nodes)

        # 2. Reverse Message Passing
        processed_edge_index, processed_edge_attr = self._reverse_mp(
            flat_edge_index, flat_edge_attr
        )

        # Proyección inicial de nodos
        x = self.node_proj(x)
        x = functional_interface.relu(x)

        curr_edge_index = processed_edge_index
        curr_edge_attr = processed_edge_attr

        # 3. Neighborhood Aggregation y Edge Updates (emlps)
        for i in range(self.num_layers):
            conv = self.convs[i]
            edge_mlp = self.edge_mlps[i]

            # PNA Conv
            x_new = conv(x, curr_edge_index, curr_edge_attr)
            x_new = functional_interface.relu(x_new)
            x_new = x_new + x
            x = functional_interface.dropout(x_new, p=self.dropout, training=self.training)

            # Edge Updates (emlps)
            src_nodes = curr_edge_index[0]
            dst_nodes = curr_edge_index[1]
            x_src = x[src_nodes]
            x_dst = x[dst_nodes]

            mlp_in = torch.cat([curr_edge_attr, x_src, x_dst], dim=-1)
            new_edge_attr = edge_mlp(mlp_in)
            curr_edge_attr = functional_interface.relu(new_edge_attr)

        return typing.cast(torch.Tensor, x)


class EdgeClassifier(nn.Module):
    """Clasificador de aristas (transacciones) usando embeddings de nodos y atributos."""

    def __init__(
        self, node_emb_dim: int, edge_attr_dim: int, hidden_dim: int, final_dropout: float = 0.1
    ) -> None:
        """Inicializa el MLP de clasificación."""
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
        """Predice la probabilidad (logits) de fraude para las aristas dadas."""
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]

        z_src = z[src_nodes]
        z_dst = z[dst_nodes]

        edge_feat = torch.cat([z_src, z_dst, edge_attr], dim=-1)

        out = self.mlp(edge_feat).squeeze(-1)
        return typing.cast(torch.Tensor, out)
