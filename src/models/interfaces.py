"""Abstract contracts/interfaces for fraud detection models."""

from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import Data


class ITraditionalModel(ABC):
    """Universal contract for traditional ML classification models."""

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the model on the provided training data."""

    @abstractmethod
    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Returns predictions for the given input features."""

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluates against ground truth and returns structured metrics."""

    @abstractmethod
    def get_underlying_model(self) -> object:
        """Returns the wrapped model object for serialization and MLflow logging."""


class IGraphModel(ABC):
    """Universal contract for Graph Neural Network classification models."""

    @abstractmethod
    def train(self, graph_data: Data) -> None:
        """Fits the GNN model on the provided graph data."""

    @abstractmethod
    def predict(self, graph_data: Data, stage: str = "val") -> np.ndarray:
        """Returns GNN prediction probabilities for the given graph and stage."""

    @abstractmethod
    def evaluate(self, graph_data: Data, stage: str = "val") -> dict[str, float]:
        """Evaluates GNN predictions and returns structured metrics."""

    @abstractmethod
    def get_underlying_model(self) -> object:
        """Returns the wrapped GNN encoder and classifier for serialization."""
