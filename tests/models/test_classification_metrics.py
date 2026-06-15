"""Unit tests for ClassificationMetricsMixin following the ZOMBIES pattern."""

import numpy as np
import pytest

from src.models.classification_metrics import ClassificationMetricsMixin


def test_zero_samples() -> None:
    """Zero: Verify that empty arrays raise ValueError."""
    y_true: np.ndarray = np.array([], dtype=np.int64)
    y_pred: np.ndarray = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):
        ClassificationMetricsMixin.compute_classification_metrics(y_true, y_pred)


def test_one_sample_correct_positive() -> None:
    """One: Test metric computation with a single correct positive sample."""
    y_true: np.ndarray = np.array([1], dtype=np.int64)
    y_pred: np.ndarray = np.array([1], dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_one_sample_incorrect_positive() -> None:
    """One: Test metric computation with a single incorrect positive sample."""
    y_true: np.ndarray = np.array([1], dtype=np.int64)
    y_pred: np.ndarray = np.array([0], dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_one_sample_correct_negative() -> None:
    """One: Test metric computation with a single correct negative sample."""
    y_true: np.ndarray = np.array([0], dtype=np.int64)
    y_pred: np.ndarray = np.array([0], dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_many_samples() -> None:
    """Many: Verify metrics with multiple samples of mixed outcomes."""
    y_true: np.ndarray = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)
    y_pred: np.ndarray = np.array([0, 1, 0, 0, 1, 1, 1, 0], dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    # y_true has 5 ones and 3 zeros.
    # True Positives (TP): predictions matching 1. indices: 1, 4, 6. Total = 3.
    # False Positives (FP): predictions = 1 but true = 0. index: 5. Total = 1.
    # False Negatives (FN): predictions = 0 but true = 1. indices: 2, 7. Total = 2.
    # True Negatives (TN): predictions = 0 but true = 0. indices: 0, 3. Total = 2.
    # Total samples = 8. Correct = 5 (indices 0, 1, 3, 4, 6).
    # accuracy = 5/8 = 0.625
    # precision = TP / (TP + FP) = 3 / 4 = 0.75
    # recall = TP / (TP + FN) = 3 / 5 = 0.6
    # f1 = 2 * (0.75 * 0.6) / (0.75 + 0.6) = 0.9 / 1.35 = 2/3 ≈ 0.66666667
    assert metrics["accuracy"] == 0.625
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.6
    assert pytest.approx(metrics["f1"]) == 2.0 / 3.0


def test_boundary_all_positive_correct() -> None:
    """Boundary: Test case where all samples are positive and correctly predicted."""
    y_true: np.ndarray = np.ones(10, dtype=np.int64)
    y_pred: np.ndarray = np.ones(10, dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_boundary_all_negative_correct() -> None:
    """Boundary: Test case where all samples are negative and correctly predicted."""
    y_true: np.ndarray = np.zeros(10, dtype=np.int64)
    y_pred: np.ndarray = np.zeros(10, dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_exception_dimension_mismatch() -> None:
    """Exception: Test that shape mismatch raises ValueError."""
    y_true: np.ndarray = np.array([0, 1, 0], dtype=np.int64)
    y_pred: np.ndarray = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        ClassificationMetricsMixin.compute_classification_metrics(y_true, y_pred)


def test_simple_happy_path() -> None:
    """Simple: Test standard correct execution with known labels."""
    y_true: np.ndarray = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    y_pred: np.ndarray = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    metrics: dict[str, float] = ClassificationMetricsMixin.compute_classification_metrics(
        y_true, y_pred
    )
    # accuracy: 4/5 = 0.8
    # precision: 2/2 = 1.0
    # recall: 2/3 ≈ 0.66666667
    # f1: 2 * (1.0 * 2/3) / (1.0 + 2/3) = 0.8
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 1.0
    assert pytest.approx(metrics["recall"]) == 2.0 / 3.0
    assert pytest.approx(metrics["f1"]) == 0.8
