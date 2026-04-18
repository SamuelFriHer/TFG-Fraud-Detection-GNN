"""Abstract interface for all classification models in the project."""

from abc import ABC, abstractmethod
from typing import Any


class IClassificationModel(ABC):
    """Universal contract for traditional ML and GNN classification models."""

    @abstractmethod
    def train(self, x_train: Any, y_train: Any) -> None:
        """Fits the model on the provided training data."""

    @abstractmethod
    def predict(self, x_input: Any) -> Any:
        """Returns predictions for the given input features."""

    @abstractmethod
    def evaluate(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """Evaluates against ground truth and returns structured metrics."""

    @abstractmethod
    def get_underlying_model(self) -> Any:
        """Returns the wrapped model object for serialization and MLflow logging."""
