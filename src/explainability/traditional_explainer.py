"""Traditional ML explainability implementation using SHAP."""

import logging

import numpy as np
import shap

from src.explainability.interfaces import ITraditionalExplainer, TraditionalExplanation
from src.models.interfaces import ITraditionalModel

logger = logging.getLogger(__name__)


class ShapExplainer(ITraditionalExplainer):
    """SHAP-based explainer for traditional machine learning models."""

    def __init__(self, model: ITraditionalModel, x_background: np.ndarray | None = None) -> None:
        """Initializes the SHAP explainer with a trained model."""
        self.underlying_model = model.get_underlying_model()

        try:
            # TreeExplainer is fast and exact for tree-based models (XGBoost, LightGBM, RF)
            self.explainer = shap.TreeExplainer(self.underlying_model)
            logger.info("Initialized shap.TreeExplainer successfully.")
        except Exception:
            logger.warning("TreeExplainer failed. Falling back to KernelExplainer/Explainer.")
            if x_background is not None:
                # Limit background size for performance with KernelExplainer
                bg_data = (
                    x_background if len(x_background) <= 100 else shap.sample(x_background, 100)
                )
                self.explainer = shap.Explainer(self.underlying_model.predict, bg_data)
            else:
                self.explainer = shap.Explainer(self.underlying_model)

    def explain_instances(
        self, x_input: np.ndarray, feature_names: list[str]
    ) -> TraditionalExplanation:
        """Computes SHAP values for the given instances."""
        logger.info("Computing SHAP values for %d instances.", len(x_input))
        shap_values_raw = self.explainer.shap_values(x_input)
        expected_value_raw = self.explainer.expected_value

        # Handle multiclass outputs for binary classification (extract positive class)
        shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw

        expected_value = 0.0
        if expected_value_raw is not None:
            if isinstance(expected_value_raw, (list, np.ndarray)):
                expected_value = float(
                    expected_value_raw[1] if len(expected_value_raw) > 1 else expected_value_raw[0]
                )
            else:
                expected_value = float(expected_value_raw)

        return TraditionalExplanation(
            shap_values=np.array(shap_values),
            expected_value=expected_value,
            feature_names=feature_names,
        )
