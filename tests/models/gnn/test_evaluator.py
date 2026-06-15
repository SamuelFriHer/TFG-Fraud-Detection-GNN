"""Unit tests for GNN prediction evaluator and threshold optimization."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from src.models.gnn.evaluator import find_optimal_threshold, get_labels_for_stage


def test_find_optimal_threshold_perfect_separation() -> None:
    """Verify threshold optimization when probabilities perfectly separate classes."""
    y_true: np.ndarray = np.array([0, 0, 1, 1], dtype=np.int32)
    probs: np.ndarray = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    best_th: float
    best_f1: float
    best_th, best_f1 = find_optimal_threshold(y_true, probs)

    assert best_th == pytest.approx(0.8)
    assert best_f1 == pytest.approx(1.0)


def test_find_optimal_threshold_imperfect_separation() -> None:
    """Verify that optimal threshold maximizes F1 score for imperfectly separated data."""
    y_true: np.ndarray = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
    probs: np.ndarray = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9], dtype=np.float32)

    best_th: float
    best_f1: float
    best_th, best_f1 = find_optimal_threshold(y_true, probs)

    max_f1: float = 0.0
    for th in probs:
        th_val: float = float(th)
        preds: np.ndarray = (probs > th_val).astype(int)
        score: float = float(f1_score(y_true, preds, zero_division=0))
        if score > max_f1:
            max_f1 = score

        preds_ge: np.ndarray = (probs >= th_val).astype(int)
        score_ge: float = float(f1_score(y_true, preds_ge, zero_division=0))
        if score_ge > max_f1:
            max_f1 = score_ge

    assert best_f1 == pytest.approx(max_f1)


def test_find_optimal_threshold_zero_division() -> None:
    """Ensure threshold search runs without errors when no positive labels exist."""
    y_true: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int32)
    probs: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    best_th: float
    best_f1: float
    best_th, best_f1 = find_optimal_threshold(y_true, probs)

    assert isinstance(best_th, float)
    assert best_f1 == pytest.approx(0.0)


def test_find_optimal_threshold_all_ones() -> None:
    """Ensure threshold search runs without errors when all labels are positive."""
    y_true: np.ndarray = np.array([1, 1, 1, 1], dtype=np.int32)
    probs: np.ndarray = np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float32)

    best_th: float
    best_f1: float
    best_th, best_f1 = find_optimal_threshold(y_true, probs)

    assert isinstance(best_th, float)
    assert best_f1 > 0.0


def test_get_labels_for_stage_valid_stages() -> None:
    """Verify get_labels_for_stage correctly extracts labels for train, val, and test stages."""
    mock_graph: MagicMock = MagicMock(spec=Data)
    mock_graph.y = torch.tensor([10, 20, 30], dtype=torch.int32)
    mock_graph.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_graph.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_graph.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

    train_labels: np.ndarray = get_labels_for_stage(mock_graph, "train")
    val_labels: np.ndarray = get_labels_for_stage(mock_graph, "val")
    test_labels: np.ndarray = get_labels_for_stage(mock_graph, "test")

    np.testing.assert_array_equal(train_labels, np.array([10], dtype=np.int32))
    np.testing.assert_array_equal(val_labels, np.array([20], dtype=np.int32))
    np.testing.assert_array_equal(test_labels, np.array([30], dtype=np.int32))


def test_get_labels_for_stage_invalid_stage() -> None:
    """Verify get_labels_for_stage raises ValueError when an invalid stage is provided."""
    mock_graph: MagicMock = MagicMock(spec=Data)
    mock_graph.y = torch.tensor([10, 20, 30], dtype=torch.int32)
    mock_graph.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_graph.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_graph.test_mask = torch.tensor([False, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="Unknown stage: invalid_stage"):
        get_labels_for_stage(mock_graph, "invalid_stage")
