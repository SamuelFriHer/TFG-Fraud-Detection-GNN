"""Visualizers for explainability outputs."""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import shap
from torch_geometric.data import Data

from src.explainability.interfaces import GraphExplanation, TraditionalExplanation

logger = logging.getLogger(__name__)


class TraditionalVisualizer:
    """Generates visualizations for traditional ML explanations."""

    @staticmethod
    def save_summary_plot(explanation: TraditionalExplanation, output_path: str) -> None:
        """Generates and saves a SHAP summary plot."""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            explanation.shap_values,
            feature_names=explanation.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to %s", output_path)

    @staticmethod
    def save_bar_plot(explanation: TraditionalExplanation, output_path: str) -> None:
        """Generates and saves a SHAP bar plot."""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            explanation.shap_values,
            feature_names=explanation.feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP bar plot to %s", output_path)


class GnnVisualizer:
    """Generates visualizations for GNN explanations."""

    @staticmethod
    def _create_explanation_graph(
        explanation: GraphExplanation,
        graph_data: Data,
        top_k_edges: int,
    ) -> nx.Graph | None:
        """Creates a NetworkX graph from the top K explanation edges."""
        if explanation.edge_mask is None:
            return None

        edge_mask = explanation.edge_mask.numpy()
        top_indices = edge_mask.argsort()[-top_k_edges:][::-1]

        edge_index = graph_data.edge_index[:, top_indices]
        edge_weights = edge_mask[top_indices]

        g = nx.Graph()
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()

        for i in range(len(src_nodes)):
            g.add_edge(src_nodes[i], dst_nodes[i], weight=edge_weights[i])

        return g

    @staticmethod
    def _draw_and_save_graph(g: nx.Graph, output_path: str, top_k_edges: int) -> None:
        """Draws and saves the NetworkX graph to a file."""
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(g)

        nx.draw_networkx_nodes(g, pos, node_size=300, node_color="lightblue")

        edges = g.edges(data=True)
        weights = [d["weight"] * 5.0 for u, v, d in edges]  # Scale width
        nx.draw_networkx_edges(g, pos, width=weights, edge_color="gray", alpha=0.7)
        nx.draw_networkx_labels(g, pos, font_size=10)

        plt.title(f"GNN Explanation Subgraph (Top {top_k_edges} edges)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved GNN subgraph plot to %s", output_path)

    @staticmethod
    def save_subgraph_plot(
        explanation: GraphExplanation,
        graph_data: Data,
        output_path: str,
        top_k_edges: int = 20,
    ) -> None:
        """Saves a plot of the most important edges from the GNN explanation."""
        g = GnnVisualizer._create_explanation_graph(explanation, graph_data, top_k_edges)

        if g is None:
            logger.warning("No edge mask found in explanation. Cannot plot subgraph.")
            return

        GnnVisualizer._draw_and_save_graph(g, output_path, top_k_edges)
