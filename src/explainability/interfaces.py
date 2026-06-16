"""Abstract contracts and data structures for explainability models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from src.models.interfaces import IGraphModel, ITraditionalModel


@dataclass
class TraditionalExplanation:
    """Standardized output for traditional ML explanations."""

    shap_values: np.ndarray
    expected_value: float | np.ndarray
    feature_names: list[str]


@dataclass
class GraphExplanation:
    """Standardized output for GNN explanations."""

    node_mask: torch.Tensor | None
    edge_mask: torch.Tensor | None
    target: int | None


class ITraditionalExplainer(ABC):
    """Universal contract for traditional ML explainers."""

    @abstractmethod
    def __init__(self, model: ITraditionalModel, x_background: np.ndarray | None = None) -> None:
        """Initializes the explainer with a trained traditional model."""

    @abstractmethod
    def explain_instances(
        self, x_input: np.ndarray, feature_names: list[str]
    ) -> TraditionalExplanation:
        """Explains the given instances and returns standardized values."""


class IGraphExplainer(ABC):
    """Universal contract for Graph Neural Network explainers."""

    @abstractmethod
    def __init__(self, model: IGraphModel) -> None:
        """Initializes the explainer with a trained GNN model."""

    @abstractmethod
    def explain_graph(
        self, graph_data: Data, index: int | None = None, target: int | None = None
    ) -> GraphExplanation:
        """Explains the predictions on the given graph data."""
