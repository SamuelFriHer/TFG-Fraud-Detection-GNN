"""Traditional ML model sub-package with factory registry."""

from typing import Any

from src.models.interfaces import ITraditionalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.svm_model import SVMModel
from src.models.traditional.xgboost_model import XGBoostModel

MODEL_REGISTRY: dict[str, type[ITraditionalModel]] = {
    "XGBoost": XGBoostModel,
    "RandomForest": RandomForestModel,
    "LightGBM": LightGBMModel,
    "SVM": SVMModel,
}

ALL_MODEL_NAMES: list[str] = list(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs: Any) -> ITraditionalModel:
    """Factory function to instantiate a model by name with optional hyperparameters."""
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(ALL_MODEL_NAMES)
        raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")
    return MODEL_REGISTRY[model_name](**kwargs)
