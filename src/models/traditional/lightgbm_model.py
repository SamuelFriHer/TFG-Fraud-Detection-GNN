"""LightGBM classifier implementing IClassificationModel."""

import logging
import warnings
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.base import IClassificationModel
from src.utils.gpu_availability import GpuAvailabilityChecker


class LightGBMModel(IClassificationModel):
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

    def train(self, x_train: Any, y_train: Any) -> None:
        """Fits the LightGBM classifier on training data."""
        self.logger.info("Training LightGBM Classifier...")
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: Any) -> np.ndarray:
        """Returns class predictions for the given input."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            predictions = self.classifier.predict(x_input)
            self._handle_prediction_warnings(caught_warnings)
        return np.asarray(predictions)

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
