"""XGBoost classifier implementing IClassificationModel."""

import logging
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.models.classification_metrics import ClassificationMetricsMixin
from src.models.interfaces import ITraditionalModel
from src.utils.gpu_availability import GpuAvailabilityChecker


class XGBoostModel(ITraditionalModel, ClassificationMetricsMixin):
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

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the XGBoost classifier on training data."""
        self.logger.info("Training XGBoost Classifier...")
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Returns class predictions for the given input."""
        return np.asarray(self.classifier.predict(x_input))

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 against test labels."""
        predictions = self.predict(x_test)
        metrics = self.compute_classification_metrics(y_test, predictions)
        self.logger.info("XGBoost metrics: %s", metrics)
        return metrics

    def get_underlying_model(self) -> object:
        """Returns the XGBClassifier instance."""
        return self.classifier
