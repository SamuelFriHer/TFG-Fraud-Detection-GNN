"""LightGBM classifier implementing IClassificationModel."""

import logging
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.base import IClassificationModel


class LightGBMModel(IClassificationModel):
    """LightGBM gradient boosting classifier."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes LightGBM with sensible defaults, overridable via kwargs."""
        defaults: dict[str, Any] = {"random_state": 42}
        defaults.update(kwargs)
        self.classifier = LGBMClassifier(**defaults)
        self.logger = logging.getLogger(__name__)

    def train(self, x_train: Any, y_train: Any) -> None:
        """Fits the LightGBM classifier on training data."""
        self.logger.info("Training LightGBM Classifier...")
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: Any) -> Any:
        """Returns class predictions for the given input."""
        return self.classifier.predict(x_input)

    def evaluate(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 against test labels."""
        predictions = self.predict(x_test)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        self.logger.info("LightGBM metrics: %s", metrics)
        return metrics

    def get_underlying_model(self) -> Any:
        """Returns the LGBMClassifier instance."""
        return self.classifier
