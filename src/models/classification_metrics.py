"""Shared classification metric computation for all model implementations."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationMetricsMixin:
    """Provides reusable binary classification metric computation."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray, y_predicted: np.ndarray
    ) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 for binary classification."""
        return {
            "accuracy": float(accuracy_score(y_true, y_predicted)),
            "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
            "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
            "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        }
