"""SVM classifier with cuML GPU acceleration and sklearn fallback."""

import logging
from typing import Any

import numpy as np

from src.models.base import IClassificationModel
from src.models.classification_metrics import ClassificationMetricsMixin
from src.utils.gpu_availability import GpuAvailabilityChecker

_CUML_SVC_SUPPORTED_PARAMS: set[str] = {
    "C",
    "kernel",
    "degree",
    "gamma",
    "coef0",
    "tol",
    "cache_size",
    "max_iter",
    "verbose",
    "probability",
}


class SVMModel(IClassificationModel, ClassificationMetricsMixin):
    """SVM classifier using cuML (GPU) when available, sklearn (CPU) otherwise."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes SVM with the best available backend."""
        self.logger = logging.getLogger(__name__)
        defaults: dict[str, Any] = {"random_state": 42, "verbose": True}
        defaults.update(kwargs)

        if GpuAvailabilityChecker().is_cuml_available():
            self.classifier, self._backend = self._build_cuml_classifier(defaults)
        else:
            self.classifier, self._backend = self._build_sklearn_classifier(defaults)

        self.logger.info("SVM backend: %s", self._backend)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the SVM classifier on training data."""
        self.logger.info("Training SVM (%s)...", self._backend)
        self.classifier.fit(x_train, y_train)

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Returns class predictions for the given input."""
        return np.asarray(self.classifier.predict(x_input))

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Computes accuracy, precision, recall, and F1 against test labels."""
        predictions = self.predict(x_test)
        metrics = self.compute_classification_metrics(y_test, predictions)
        self.logger.info("SVM metrics: %s", metrics)
        return metrics

    def get_underlying_model(self) -> object:
        """Returns the underlying classifier instance."""
        return self.classifier

    @staticmethod
    def _build_cuml_classifier(params: dict[str, Any]) -> tuple[Any, str]:
        """Constructs a cuML SVC, filtering out unsupported params like random_state."""
        from cuml.svm import SVC  # type: ignore

        cuml_params = {k: v for k, v in params.items() if k in _CUML_SVC_SUPPORTED_PARAMS}
        return SVC(**cuml_params), "cuML (GPU)"

    @staticmethod
    def _build_sklearn_classifier(params: dict[str, Any]) -> tuple[Any, str]:
        """Constructs a scikit-learn SVC with all params."""
        from sklearn.svm import SVC  # type: ignore

        return SVC(**params), "scikit-learn (CPU)"
