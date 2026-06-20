"""Unit tests for the explainability interfaces and dataclasses."""

import numpy as np
import torch

from src.explainability.interfaces import GraphExplanation, TraditionalExplanation


def test_traditional_explanation_instantiation() -> None:
    """Verifies that TraditionalExplanation instantiates with correct values."""
    shap_values_array: np.ndarray = np.array([0.1, -0.2, 0.5])
    expected_val: float = 0.55
    feature_names_list: list[str] = ["feat1", "feat2", "feat3"]

    explanation_instance: TraditionalExplanation = TraditionalExplanation(
        shap_values=shap_values_array,
        expected_value=expected_val,
        feature_names=feature_names_list,
    )

    assert np.array_equal(explanation_instance.shap_values, shap_values_array)
    assert explanation_instance.expected_value == expected_val
    assert explanation_instance.feature_names == feature_names_list


def test_graph_explanation_instantiation() -> None:
    """Verifies that GraphExplanation instantiates with correct values."""
    node_mask_tensor: torch.Tensor = torch.tensor([1.0, 0.0])
    edge_mask_tensor: torch.Tensor = torch.tensor([0.5, 0.8])
    target_idx: int = 1

    explanation_instance: GraphExplanation = GraphExplanation(
        node_mask=node_mask_tensor,
        edge_mask=edge_mask_tensor,
        target=target_idx,
    )

    assert torch.equal(explanation_instance.node_mask, node_mask_tensor)
    assert torch.equal(explanation_instance.edge_mask, edge_mask_tensor)
    assert explanation_instance.target == target_idx
