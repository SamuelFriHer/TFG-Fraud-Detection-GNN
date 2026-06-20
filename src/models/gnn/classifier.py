"""Definition of the edge-level transaction classifier."""

import typing

import torch
from torch import nn


class EdgeClassifier(nn.Module):
    """Edge (transaction) classifier using node embeddings and edge attributes."""

    def __init__(
        self,
        node_emb_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        final_dropout: float = 0.1,
    ) -> None:
        """Initializes the classification MLP."""
        super().__init__()
        in_dim: int = (node_emb_dim * 2) + edge_attr_dim

        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=final_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=final_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the probability (logits of fraud) for the given edges."""
        edge_feat: torch.Tensor = torch.cat(
            [z[edge_label_index[0]], z[edge_label_index[1]], edge_attr], dim=-1
        )
        out: torch.Tensor = self.mlp(edge_feat).squeeze(-1)
        return typing.cast(torch.Tensor, out)
