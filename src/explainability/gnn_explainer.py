"""GNN explainability implementation using PyTorch Geometric."""

import logging
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from src.explainability.interfaces import GraphExplanation, IGraphExplainer
from src.models.interfaces import IGraphModel

logger = logging.getLogger(__name__)


class EdgeClassificationWrapper(nn.Module):
    """Wraps the encoder and classifier for edge-level GNNExplainer."""

    def __init__(self, encoder: nn.Module, classifier: nn.Module) -> None:
        """Initializes the wrapper."""
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the whole network."""
        z = self.encoder(x, edge_index, edge_attr)
        logits = self.classifier(z, edge_index, edge_attr)
        return logits


class GnnExplainerModel(IGraphExplainer):
    """GNNExplainer implementation for edge-level fraud detection models."""

    def __init__(self, model: IGraphModel) -> None:
        """Initializes the GNN explainer with a trained GNN model."""
        encoder, classifier = model.get_underlying_model()  # type: ignore
        self.wrapped_model = EdgeClassificationWrapper(encoder, classifier)  # type: ignore
        self.wrapped_model.eval()

        self.explainer = Explainer(
            model=self.wrapped_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="edge",
                return_type="raw",
            ),
        )
        logger.info("Initialized PyTorch Geometric GNNExplainer for edge classification.")

    def _prepare_node_features(
        self, x: torch.Tensor, edge_index: torch.Tensor, index: int
    ) -> torch.Tensor:
        """Injects ego-ID flags if the input dimension is missing it."""
        expected_dim = self.wrapped_model.encoder.node_proj.in_features
        if x.size(-1) == expected_dim - 1:
            num_nodes = x.size(0)
            ego_flag = x.new_zeros((num_nodes, 1))
            target_edge = edge_index[:, index]
            ego_flag[target_edge[0]] = 1.0
            ego_flag[target_edge[1]] = 1.0
            return torch.cat([x, ego_flag], dim=-1)
        return x

    def explain_graph(
        self, graph_data: Data, index: int | None = None, target: int | None = None
    ) -> GraphExplanation:
        """Computes explanation masks for the given graph and index."""
        if index is None:
            raise ValueError("index (edge_index) is required for explanation.")

        logger.info("Computing GNN explanation for index %d.", index)

        target_tensor = torch.tensor([target]) if target is not None else None
        device = next(self.wrapped_model.parameters()).device
        if target_tensor is not None:
            target_tensor = target_tensor.to(device)

        x = self._prepare_node_features(graph_data.x.to(device), graph_data.edge_index, index)

        explanation: Any = self.explainer(
            x=x,
            edge_index=graph_data.edge_index.to(device),
            edge_attr=graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None,
            index=index,
            target=target_tensor,
        )

        n_mask = explanation.node_mask
        e_mask = explanation.edge_mask

        return GraphExplanation(
            node_mask=n_mask.detach().cpu() if n_mask is not None else None,
            edge_mask=e_mask.detach().cpu() if e_mask is not None else None,
            target=target,
        )
