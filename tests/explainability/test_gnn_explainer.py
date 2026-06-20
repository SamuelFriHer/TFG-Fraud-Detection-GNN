"""Unit tests for the GNN explainability implementations."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from src.explainability.gnn_explainer import EdgeClassificationWrapper, GnnExplainerModel
from src.models.interfaces import IGraphModel


def test_edge_classification_wrapper_forward() -> None:
    """Verifies the forward pass of EdgeClassificationWrapper."""
    mock_encoder = MagicMock()
    mock_classifier = MagicMock()

    wrapper_instance = EdgeClassificationWrapper(mock_encoder, mock_classifier)

    x_tensor = torch.randn(5, 10)
    edge_index_tensor = torch.tensor([[0, 1], [1, 2]])
    edge_attr_tensor = torch.randn(2, 3)

    mock_encoder.return_value = torch.randn(5, 8)
    mock_classifier.return_value = torch.randn(2, 1)

    result_logits = wrapper_instance(x_tensor, edge_index_tensor, edge_attr_tensor)

    mock_encoder.assert_called_once_with(x_tensor, edge_index_tensor, edge_attr_tensor)
    mock_classifier.assert_called_once_with(
        mock_encoder.return_value, edge_index_tensor, edge_attr_tensor
    )
    assert torch.equal(result_logits, mock_classifier.return_value)


def test_prepare_node_features_with_ego_flag() -> None:
    """Verifies ego flag injection when node features dimension is missing one column."""
    mock_encoder = MagicMock()
    mock_encoder.node_proj.in_features = 11  # Expected dim is 11
    mock_classifier = MagicMock()

    mock_model = MagicMock(spec=IGraphModel)
    mock_model.get_underlying_model.return_value = (mock_encoder, mock_classifier)

    with patch("src.explainability.gnn_explainer.Explainer"):
        explainer_instance = GnnExplainerModel(mock_model)

        # Test Case: x has shape (num_nodes, 10) -> missing 1 feature
        x_tensor = torch.zeros((4, 10))
        edge_index_tensor = torch.tensor([[0, 1, 2], [1, 2, 3]])
        # index is 1, which corresponds to edge (1, 2)
        target_index = 1

        prepared_features = explainer_instance._prepare_node_features(
            x_tensor, edge_index_tensor, target_index
        )

        # Dim should become 11
        assert prepared_features.size(-1) == 11
        # Ego flag should be 1.0 for node 1 and node 2 (endpoints of edge_index[:, 1])
        assert prepared_features[0, -1] == 0.0
        assert prepared_features[1, -1] == 1.0
        assert prepared_features[2, -1] == 1.0
        assert prepared_features[3, -1] == 0.0


def test_prepare_node_features_without_ego_flag() -> None:
    """Verifies that features are unmodified when node feature dimension matches expected."""
    mock_encoder = MagicMock()
    mock_encoder.node_proj.in_features = 11
    mock_classifier = MagicMock()

    mock_model = MagicMock(spec=IGraphModel)
    mock_model.get_underlying_model.return_value = (mock_encoder, mock_classifier)

    with patch("src.explainability.gnn_explainer.Explainer"):
        explainer_instance = GnnExplainerModel(mock_model)

        x_tensor = torch.zeros((4, 11))
        edge_index_tensor = torch.tensor([[0, 1, 2], [1, 2, 3]])

        prepared_features = explainer_instance._prepare_node_features(
            x_tensor, edge_index_tensor, 1
        )

        # Should be returned unmodified
        assert prepared_features.size(-1) == 11
        assert torch.equal(prepared_features, x_tensor)


def test_explain_graph_missing_index() -> None:
    """Verifies that ValueError is raised if index is None."""
    mock_encoder = MagicMock()
    mock_classifier = MagicMock()
    mock_model = MagicMock(spec=IGraphModel)
    mock_model.get_underlying_model.return_value = (mock_encoder, mock_classifier)

    with patch("src.explainability.gnn_explainer.Explainer"):
        explainer_instance = GnnExplainerModel(mock_model)
        graph_data_mock = MagicMock(spec=Data)

        with pytest.raises(ValueError, match="index \\(edge_index\\) is required"):
            explainer_instance.explain_graph(graph_data_mock, index=None)


def test_explain_graph_workflow() -> None:
    """Verifies the explain_graph workflow with mocked PyG explainer."""
    mock_encoder = MagicMock()
    mock_encoder.node_proj.in_features = 10
    mock_classifier = MagicMock()
    mock_model = MagicMock(spec=IGraphModel)
    mock_model.get_underlying_model.return_value = (mock_encoder, mock_classifier)

    with patch("src.explainability.gnn_explainer.Explainer") as mock_explainer_class:
        mock_explainer_instance = MagicMock()
        mock_explainer_class.return_value = mock_explainer_instance

        explainer_instance = GnnExplainerModel(mock_model)
        # Mock parameters iterator so next(parameters()) resolves device
        explainer_instance.wrapped_model.parameters = MagicMock(
            return_value=iter([torch.tensor([1.0])])
        )

        # Prepare inputs
        x_tensor = torch.randn(4, 10)
        edge_index_tensor = torch.tensor([[0, 1], [1, 2]])
        edge_attr_tensor = torch.randn(2, 3)
        graph_data_input = Data(
            x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor
        )

        # Mock explanation returned by PyG Explainer
        explanation_result_mock = MagicMock()
        explanation_result_mock.node_mask = torch.tensor([0.1, 0.2])
        explanation_result_mock.edge_mask = torch.tensor([0.3, 0.4])
        mock_explainer_instance.return_value = explanation_result_mock

        explanation_output = explainer_instance.explain_graph(graph_data_input, index=1, target=0)

        # Assertions
        mock_explainer_instance.assert_called_once()
        call_kwargs = mock_explainer_instance.call_args[1]

        assert torch.equal(call_kwargs["x"], x_tensor)
        assert torch.equal(call_kwargs["edge_index"], edge_index_tensor)
        assert torch.equal(call_kwargs["edge_attr"], edge_attr_tensor)
        assert call_kwargs["index"] == 1
        assert torch.equal(call_kwargs["target"], torch.tensor([0]))

        assert torch.equal(explanation_output.node_mask, explanation_result_mock.node_mask)
        assert torch.equal(explanation_output.edge_mask, explanation_result_mock.edge_mask)
        assert explanation_output.target == 0
