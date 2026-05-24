"""Definición de las capas y arquitecturas de red neuronal de grafos (GNN)."""

import typing

import torch
import torch.nn.functional as functional_interface
from torch import nn
from torch_geometric.nn import GATv2Conv  # type: ignore


class GATv2Encoder(nn.Module):
    """Codificador de nodos basado en GATv2.

    Aprende embeddings de los nodos agregando información
    de sus vecinos e incorporando atributos de arista.
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 4,
        concat: bool = True,
    ) -> None:
        """Inicializa las capas convolucionales GATv2."""
        super().__init__()
        self.convs = nn.ModuleList()
        self.concat = concat
        self.heads = heads

        self.convs.append(
            GATv2Conv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=concat,
                edge_dim=edge_dim,
            )
        )

        out_dim = hidden_channels * heads if concat else hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    out_dim,
                    hidden_channels,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                )
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Realiza el paso forward a través de las capas GATv2."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i != len(self.convs) - 1:
                x = x.relu()
                x = functional_interface.dropout(x, p=0.5, training=self.training)
        return typing.cast(torch.Tensor, x)


class EdgeClassifier(nn.Module):
    """Clasificador de aristas (transacciones) usando embeddings de nodos y atributos."""

    def __init__(self, node_emb_dim: int, edge_attr_dim: int, hidden_dim: int) -> None:
        """Inicializa el MLP de clasificación."""
        super().__init__()
        in_dim = (node_emb_dim * 2) + edge_attr_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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
