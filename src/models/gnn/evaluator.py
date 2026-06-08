"""Evaluation utilities for GNN models."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch_geometric.data import Data

from src.models.classification_metrics import ClassificationMetricsMixin


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Finds the decision threshold that maximizes the F1-Score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_th, float(f1_scores[best_idx])


def evaluate_predictions(probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Evaluates GNN predictions and optimizes the threshold for F1-Score."""
    threshold, best_f1 = find_optimal_threshold(y_true, probs)
    preds = (probs > threshold).astype(int)
    metrics = ClassificationMetricsMixin.compute_classification_metrics(y_true, preds)
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "optimal_threshold": threshold,
        }
    )
    return metrics


def evaluate_predictions_at_threshold(
    probs: np.ndarray, y_true: np.ndarray, threshold: float
) -> dict[str, float]:
    """Computes classification metrics using a fixed decision threshold."""
    preds = (probs > threshold).astype(int)
    metrics = ClassificationMetricsMixin.compute_classification_metrics(y_true, preds)
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "optimal_threshold": threshold,
        }
    )
    return metrics


def get_labels_for_stage(graph_data: Data, stage: str) -> np.ndarray:
    """Extracts ground truth labels for the given stage mask."""
    mask_map = {
        "train": graph_data.train_mask,
        "val": graph_data.val_mask,
        "test": graph_data.test_mask,
    }
    if stage not in mask_map:
        raise ValueError(f"Unknown stage: {stage}")
    return graph_data.y[mask_map[stage]].cpu().numpy()
