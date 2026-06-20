"""Unit tests for the ShapExplainer class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.explainability.interfaces import TraditionalExplanation
from src.explainability.traditional_explainer import ShapExplainer
from src.models.interfaces import ITraditionalModel


@pytest.fixture
def mock_traditional_model() -> MagicMock:
    """Provides a mocked traditional model implementing ITraditionalModel."""
    model_mock = MagicMock(spec=ITraditionalModel)
    underlying_mock = MagicMock()
    model_mock.get_underlying_model.return_value = underlying_mock
    return model_mock


@patch("src.explainability.traditional_explainer.shap.TreeExplainer")
def test_shap_explainer_init_tree_explainer_success(
    mock_tree_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies that ShapExplainer prefers TreeExplainer by default."""
    explainer_instance = ShapExplainer(mock_traditional_model)

    assert explainer_instance.underlying_model == mock_traditional_model.get_underlying_model()
    mock_tree_explainer.assert_called_once_with(mock_traditional_model.get_underlying_model())
    assert explainer_instance.explainer == mock_tree_explainer.return_value


@patch("src.explainability.traditional_explainer.shap.Explainer")
@patch(
    "src.explainability.traditional_explainer.shap.TreeExplainer",
    side_effect=Exception("Tree failed"),
)
def test_shap_explainer_init_fallback_no_bg(
    mock_tree_explainer: MagicMock,
    mock_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies fallback to Explainer when TreeExplainer raises an Exception."""
    explainer_instance = ShapExplainer(mock_traditional_model)

    mock_tree_explainer.assert_called_once()
    mock_explainer.assert_called_once_with(mock_traditional_model.get_underlying_model())
    assert explainer_instance.explainer == mock_explainer.return_value


@patch("src.explainability.traditional_explainer.shap.sample")
@patch("src.explainability.traditional_explainer.shap.Explainer")
@patch(
    "src.explainability.traditional_explainer.shap.TreeExplainer",
    side_effect=Exception("Tree failed"),
)
def test_shap_explainer_init_fallback_with_bg_large(
    mock_tree_explainer: MagicMock,
    mock_explainer: MagicMock,
    mock_sample: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies fallback to Explainer with sampled background if bg size > 100."""
    x_background_large = np.random.rand(120, 5)
    sampled_bg = np.random.rand(100, 5)
    mock_sample.return_value = sampled_bg

    explainer_instance = ShapExplainer(mock_traditional_model, x_background=x_background_large)

    mock_tree_explainer.assert_called_once()
    mock_sample.assert_called_once_with(x_background_large, 100)
    mock_explainer.assert_called_once_with(
        mock_traditional_model.get_underlying_model().predict,
        sampled_bg,
    )
    assert explainer_instance.explainer == mock_explainer.return_value


@patch("src.explainability.traditional_explainer.shap.Explainer")
@patch(
    "src.explainability.traditional_explainer.shap.TreeExplainer",
    side_effect=Exception("Tree failed"),
)
def test_shap_explainer_init_fallback_with_bg_small(
    mock_tree_explainer: MagicMock,
    mock_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies fallback to Explainer using raw background if bg size <= 100."""
    x_background_small = np.random.rand(50, 5)

    explainer_instance = ShapExplainer(mock_traditional_model, x_background=x_background_small)

    mock_tree_explainer.assert_called_once()
    mock_explainer.assert_called_once_with(
        mock_traditional_model.get_underlying_model().predict,
        x_background_small,
    )
    assert explainer_instance.explainer == mock_explainer.return_value


@patch("src.explainability.traditional_explainer.shap.TreeExplainer")
def test_explain_instances_array_shap_values(
    mock_tree_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies explain_instances when shap returns a single numpy array."""
    mock_tree_instance = MagicMock()
    mock_tree_explainer.return_value = mock_tree_instance

    shap_values_mock = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_tree_instance.shap_values.return_value = shap_values_mock
    mock_tree_instance.expected_value = 0.55

    explainer_instance = ShapExplainer(mock_traditional_model)
    x_input = np.random.rand(2, 2)
    feature_names_list = ["f1", "f2"]

    explanation_instance: TraditionalExplanation = explainer_instance.explain_instances(
        x_input, feature_names_list
    )

    mock_tree_instance.shap_values.assert_called_once_with(x_input)
    assert np.array_equal(explanation_instance.shap_values, shap_values_mock)
    assert explanation_instance.expected_value == 0.55
    assert explanation_instance.feature_names == feature_names_list


@patch("src.explainability.traditional_explainer.shap.TreeExplainer")
def test_explain_instances_list_shap_values_and_list_expected(
    mock_tree_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies explain_instances when shap returns list values and expected values."""
    mock_tree_instance = MagicMock()
    mock_tree_explainer.return_value = mock_tree_instance

    shap_values_mock = [np.array([[0.1], [0.2]]), np.array([[0.3], [0.4]])]
    mock_tree_instance.shap_values.return_value = shap_values_mock
    mock_tree_instance.expected_value = [0.1, 0.9]

    explainer_instance = ShapExplainer(mock_traditional_model)
    x_input = np.random.rand(2, 1)
    feature_names_list = ["f1"]

    explanation_instance: TraditionalExplanation = explainer_instance.explain_instances(
        x_input, feature_names_list
    )

    # Positive class (index 1) should be extracted
    assert np.array_equal(explanation_instance.shap_values, shap_values_mock[1])
    assert explanation_instance.expected_value == 0.9
    assert explanation_instance.feature_names == feature_names_list


@patch("src.explainability.traditional_explainer.shap.TreeExplainer")
def test_explain_instances_list_expected_single_value(
    mock_tree_explainer: MagicMock,
    mock_traditional_model: MagicMock,
) -> None:
    """Verifies expected value extraction when expected_value is a single-element list."""
    mock_tree_instance = MagicMock()
    mock_tree_explainer.return_value = mock_tree_instance

    shap_values_mock = np.array([[0.1], [0.2]])
    mock_tree_instance.shap_values.return_value = shap_values_mock
    mock_tree_instance.expected_value = [0.45]

    explainer_instance = ShapExplainer(mock_traditional_model)
    x_input = np.random.rand(2, 1)
    feature_names_list = ["f1"]

    explanation_instance: TraditionalExplanation = explainer_instance.explain_instances(
        x_input, feature_names_list
    )

    assert explanation_instance.expected_value == 0.45
