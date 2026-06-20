"""Unit tests for the explainability visualizers."""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.explainability.interfaces import GraphExplanation, TraditionalExplanation
from src.explainability.visualizers import GnnVisualizer, TraditionalVisualizer


@patch("src.explainability.visualizers.shap")
@patch("src.explainability.visualizers.plt")
def test_traditional_visualizer_save_summary_plot(
    mock_plt: MagicMock,
    mock_shap: MagicMock,
) -> None:
    """Verifies that save_summary_plot configures and saves the summary plot correctly."""
    explanation_input = TraditionalExplanation(
        shap_values=np.array([0.1, 0.2]),
        expected_value=0.5,
        feature_names=["f1", "f2"],
    )
    output_file_path = "summary_output.png"

    TraditionalVisualizer.save_summary_plot(explanation_input, output_file_path)

    mock_plt.figure.assert_called_once_with(figsize=(10, 8))
    mock_shap.summary_plot.assert_called_once_with(
        explanation_input.shap_values,
        feature_names=explanation_input.feature_names,
        show=False,
    )
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(output_file_path, dpi=300, bbox_inches="tight")
    mock_plt.close.assert_called_once()


@patch("src.explainability.visualizers.shap")
@patch("src.explainability.visualizers.plt")
def test_traditional_visualizer_save_bar_plot(
    mock_plt: MagicMock,
    mock_shap: MagicMock,
) -> None:
    """Verifies that save_bar_plot configures and saves the bar plot correctly."""
    explanation_input = TraditionalExplanation(
        shap_values=np.array([0.1, 0.2]),
        expected_value=0.5,
        feature_names=["f1", "f2"],
    )
    output_file_path = "bar_output.png"

    TraditionalVisualizer.save_bar_plot(explanation_input, output_file_path)

    mock_plt.figure.assert_called_once_with(figsize=(10, 8))
    mock_shap.summary_plot.assert_called_once_with(
        explanation_input.shap_values,
        feature_names=explanation_input.feature_names,
        plot_type="bar",
        show=False,
    )
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(output_file_path, dpi=300, bbox_inches="tight")
    mock_plt.close.assert_called_once()


def test_gnn_visualizer_create_graph_missing_mask() -> None:
    """Verifies that _create_explanation_graph returns None when edge_mask is missing."""
    explanation_input = GraphExplanation(node_mask=None, edge_mask=None, target=1)
    graph_data_input = Data(x=torch.zeros((2, 2)), edge_index=torch.tensor([[0], [1]]))

    result_graph = GnnVisualizer._create_explanation_graph(
        explanation_input, graph_data_input, top_k_edges=5
    )

    assert result_graph is None


def test_gnn_visualizer_create_graph_success() -> None:
    """Verifies that _create_explanation_graph builds the NetworkX graph with top edges."""
    edge_mask_tensor = torch.tensor([0.1, 0.9, 0.4])
    explanation_input = GraphExplanation(node_mask=None, edge_mask=edge_mask_tensor, target=1)

    edge_index_tensor = torch.tensor([[0, 1, 2], [1, 2, 3]])
    graph_data_input = Data(x=torch.zeros((4, 2)), edge_index=edge_index_tensor)

    # Top 2 edges should be index 1 (weight 0.9) and index 2 (weight 0.4).
    # These correspond to edges (1, 2) and (2, 3).
    result_graph = GnnVisualizer._create_explanation_graph(
        explanation_input, graph_data_input, top_k_edges=2
    )

    assert isinstance(result_graph, nx.Graph)
    assert len(result_graph.nodes) == 3
    assert len(result_graph.edges) == 2
    # Verify weights
    assert result_graph.edges[1, 2]["weight"] == 0.9
    assert result_graph.edges[2, 3]["weight"] == 0.4


@patch("src.explainability.visualizers.logger")
def test_gnn_visualizer_save_subgraph_plot_none_graph(
    mock_logger: MagicMock,
) -> None:
    """Verifies that save_subgraph_plot logs a warning and returns if graph is None."""
    explanation_input = GraphExplanation(node_mask=None, edge_mask=None, target=1)
    graph_data_input = Data(x=torch.zeros((2, 2)), edge_index=torch.tensor([[0], [1]]))

    GnnVisualizer.save_subgraph_plot(explanation_input, graph_data_input, "subgraph.png")

    mock_logger.warning.assert_called_once_with(
        "No edge mask found in explanation. Cannot plot subgraph."
    )


@patch("src.explainability.visualizers.nx")
@patch("src.explainability.visualizers.plt")
def test_gnn_visualizer_save_subgraph_plot_success(
    mock_plt: MagicMock,
    mock_nx: MagicMock,
) -> None:
    """Verifies that save_subgraph_plot calls draw and save operations correctly."""
    explanation_input = GraphExplanation(node_mask=None, edge_mask=torch.tensor([0.8]), target=1)
    graph_data_input = Data(x=torch.zeros((2, 2)), edge_index=torch.tensor([[0], [1]]))
    output_file_path = "subgraph.png"

    # Mock the NetworkX graph created
    graph_instance = nx.Graph()
    graph_instance.add_edge(0, 1, weight=0.8)
    mock_nx.Graph.return_value = graph_instance
    mock_nx.spring_layout.return_value = {0: [0, 0], 1: [1, 1]}

    GnnVisualizer.save_subgraph_plot(
        explanation_input, graph_data_input, output_file_path, top_k_edges=1
    )

    mock_plt.figure.assert_called_once_with(figsize=(10, 10))
    mock_nx.spring_layout.assert_called_once_with(graph_instance)
    mock_nx.draw_networkx_nodes.assert_called_once()
    mock_nx.draw_networkx_edges.assert_called_once()
    mock_nx.draw_networkx_labels.assert_called_once()
    mock_plt.title.assert_called_once_with("GNN Explanation Subgraph (Top 1 edges)")
    mock_plt.axis.assert_called_once_with("off")
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(output_file_path, dpi=300, bbox_inches="tight")
    mock_plt.close.assert_called_once()
