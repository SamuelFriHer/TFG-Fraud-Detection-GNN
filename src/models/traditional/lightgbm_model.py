"""LightGBM classifier implementing IClassificationModel."""

import logging
import warnings
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from src.models.classification_metrics import ClassificationMetricsMixin
from src.models.interfaces import ITraditionalModel
from src.utils.gpu_availability import GpuAvailabilityChecker


class LightGBMModel(ITraditionalModel, ClassificationMetricsMixin):
    """LightGBM gradient boosting classifier with automatic GPU acceleration."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes LightGBM, enabling GPU device when CUDA is detected."""
        self.logger = logging.getLogger(__name__)
        defaults: dict[str, Any] = {"random_state": 42, "verbosity": 2}

        if "device" not in kwargs and GpuAvailabilityChecker().is_cuda_available():
            defaults["device"] = "gpu"
            self.logger.info("LightGBM will train on GPU.")

        defaults.update(kwargs)
        self.classifier = LGBMClassifier(**defaults)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the LightGBM classifier on training data."""
        self.logger.info("Training LightGBM Classifier...")
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Returns class predictions for the given input."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            predictions = self.classifier.predict(x_input)
            self._handle_prediction_warnings(caught_warnings)
        return np.asarray(predictions)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 against test labels."""
        predictions = self.predict(x_test)
        metrics = self.compute_classification_metrics(y_test, predictions)
        self.logger.info("LightGBM metrics: %s", metrics)
        return metrics

    def get_underlying_model(self) -> object:
        """Returns the LGBMClassifier instance."""
        return self.classifier

    def _handle_prediction_warnings(self, caught_warnings: list[warnings.WarningMessage]) -> None:
        """Filters out known false-positive warnings from LightGBM predictions."""
        for warning_item in caught_warnings:
            if "X does not have valid feature names" in str(warning_item.message):
                self.logger.info("Ignored LightGBM false positive: feature names warning.")
            else:
                warnings.showwarning(
                    warning_item.message,
                    warning_item.category,
                    warning_item.filename,
                    warning_item.lineno,
                )
