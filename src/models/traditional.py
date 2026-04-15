import logging
from abc import ABC, abstractmethod
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from sklearn.svm import SVC  # type: ignore
from xgboost import XGBClassifier


class ITraditionalModel(ABC):
    """
    Interface for traditional machine learning models.
    """

    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        """Trains the model on provided data."""
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> float:
        """Evaluates the model and returns accuracy."""
        pass


class XGBoostModel(ITraditionalModel):
    """XGBoost Classifier wrapper maintaining clear boundaries."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **kwargs
        )
        self.logger = logging.getLogger(__name__)

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training XGBoost Classifier...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.logger.info(f"XGBoost Accuracy: {acc}")
        self.logger.info(f"\n{classification_report(y_test, preds)}")
        return float(acc)


class RandomForestModel(ITraditionalModel):
    """Random Forest Classifier wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(random_state=42, **kwargs)
        self.logger = logging.getLogger(__name__)

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.logger.info(f"Random Forest Accuracy: {acc}")
        return float(acc)


class LightGBMModel(ITraditionalModel):
    """LightGBM Classifier wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = LGBMClassifier(random_state=42, **kwargs)
        self.logger = logging.getLogger(__name__)

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training LightGBM Classifier...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.logger.info(f"LightGBM Accuracy: {acc}")
        return float(acc)


class SVMModel(ITraditionalModel):
    """Support Vector Machine Classifier wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = SVC(random_state=42, **kwargs)
        self.logger = logging.getLogger(__name__)

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training SVM Classifier...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.logger.info(f"SVM Accuracy: {acc}")
        return float(acc)
