"""Configuration dataclass for GNN model hyperparameters and architecture."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GNNModelConfig:
    """Hyperparameters and architectural settings for the GNN Fraud Detector."""

    node_feat_dim: int
    edge_feat_dim: int
    in_channels: int | None = None
    hidden_channels: int = 64
    num_layers: int = 2
    lr: float = 0.001
    batch_size: int = 2048
    epochs: int = 80
    pos_weight: float | None = None
    dropout: float = 0.1
    final_dropout: float = 0.1
    num_neighbors: list[int] = field(default_factory=lambda: [20, 10])

    def __post_init__(self) -> None:
        """Ensures that in_channels is populated."""
        if self.in_channels is None:
            object.__setattr__(self, "in_channels", self.node_feat_dim + 1)
