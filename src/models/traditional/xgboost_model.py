"""XGBoost classifier implementing IClassificationModel."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.models.base import IClassificationModel
from src.utils.gpu_availability import GpuAvailabilityChecker


class XGBoostModel(IClassificationModel):
    """XGBoost gradient boosting classifier with automatic CUDA acceleration."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes XGBoost, enabling CUDA when a GPU is detected."""
        self.logger = logging.getLogger(__name__)
        defaults: dict[str, Any] = {"eval_metric": "logloss", "verbosity": 2}

        if "device" not in kwargs and GpuAvailabilityChecker().is_cuda_available():
            defaults["device"] = "cuda"
            self.logger.info("XGBoost will train on CUDA GPU.")

        defaults.update(kwargs)
        self.classifier = XGBClassifier(**defaults)

    def train(self, x_train: Any, y_train: Any) -> None:
        """Fits the XGBoost classifier on training data."""
        self.logger.info("Training XGBoost Classifier...")
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: Any) -> np.ndarray:
        """Returns class predictions for the given input."""
        return np.asarray(self.classifier.predict(x_input))

    def evaluate(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 against test labels."""
        predictions = self.predict(x_test)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        self.logger.info("XGBoost metrics: %s", metrics)
        return metrics

    def get_underlying_model(self) -> Any:
        """Returns the XGBClassifier instance."""
        return self.classifier
