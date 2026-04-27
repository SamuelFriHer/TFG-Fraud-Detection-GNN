"""Unit tests for SVMModel following the ZOMBIES pattern."""

import numpy as np
import pytest

from src.models.base import IClassificationModel
from src.models.traditional.svm_model import SVMModel


@pytest.fixture
def model() -> SVMModel:
    """Provides a fresh SVMModel instance."""
    return SVMModel(probability=True)


def test_interface(model: SVMModel) -> None:
    """Interface: Verify model implements the correct interface."""
    assert isinstance(model, IClassificationModel)
    assert hasattr(model, "train")
    assert hasattr(model, "predict")
    assert hasattr(model, "evaluate")
    assert hasattr(model, "get_underlying_model")


def test_zero_samples(model: SVMModel) -> None:
    """Zero: Test training with zero samples."""
    x_empty = np.array([]).reshape(0, 5)
    y_empty = np.array([])
    with pytest.raises(ValueError):
        model.train(x_empty, y_empty)


def test_one_sample(model: SVMModel) -> None:
    """One: Test training and predicting with minimal samples."""
    x_train = np.random.rand(2, 5)
    y_train = np.array([0, 1])
    model.train(x_train, y_train)

    x_input = np.random.rand(1, 5)
    prediction = model.predict(x_input)
    assert len(prediction) == 1


def test_many_samples(model: SVMModel) -> None:
    """Many: Test with a representative number of samples."""
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    model.train(x_train, y_train)

    x_test = np.random.rand(20, 10)
    predictions = model.predict(x_test)
    assert len(predictions) == 20


def test_boundary_features(model: SVMModel) -> None:
    """Boundary: Test with minimum number of features (1)."""
    x_train = np.random.rand(10, 1)
    y_train = np.random.randint(0, 2, 10)
    model.train(x_train, y_train)

    x_input = np.random.rand(5, 1)
    predictions = model.predict(x_input)
    assert len(predictions) == 5


def test_exception_invalid_features(model: SVMModel) -> None:
    """Exception: Test prediction with mismatching feature count."""
    x_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    model.train(x_train, y_train)

    x_invalid = np.random.rand(5, 3)
    with pytest.raises(ValueError):
        model.predict(x_invalid)


def test_simple_happy_path(model: SVMModel) -> None:
    """Simple: Test a basic full flow."""
    x_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, 50)
    model.train(x_train, y_train)

    metrics = model.evaluate(x_train, y_train)
    assert "accuracy" in metrics
    for val in metrics.values():
        assert 0.0 <= val <= 1.0
