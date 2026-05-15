"""Abstract interface for all classification models in the project."""

from abc import ABC, abstractmethod

import numpy as np


class IClassificationModel(ABC):
    """Universal contract for traditional ML and GNN classification models."""

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
