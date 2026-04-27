"""Unit tests for LightGBMModel following the ZOMBIES pattern."""

import numpy as np
import pandas as pd  # type: ignore
import pytest

from src.models.base import IClassificationModel
from src.models.traditional.lightgbm_model import LightGBMModel


@pytest.fixture
def model() -> LightGBMModel:
    """Provides a fresh LightGBMModel instance."""
    return LightGBMModel(n_estimators=10)


def test_interface(model: LightGBMModel) -> None:
    """Interface: Verify model implements the correct interface."""
    assert isinstance(model, IClassificationModel)
    assert hasattr(model, "train")
    assert hasattr(model, "predict")
    assert hasattr(model, "evaluate")
    assert hasattr(model, "get_underlying_model")


def test_zero_samples(model: LightGBMModel) -> None:
    """Zero: Test training with zero samples."""
    x_empty = pd.DataFrame(columns=["f1", "f2", "f3", "f4", "f5"], dtype=float)
    y_empty = np.array([])
    with pytest.raises((ValueError, Exception)):
        model.train(x_empty, y_empty)


def test_one_sample(model: LightGBMModel) -> None:
    """One: Test training and predicting with minimal samples."""
    cols = [f"f{i}" for i in range(5)]
    x_train = pd.DataFrame(np.random.rand(2, 5), columns=cols)
    y_train = np.array([0, 1])
    model.train(x_train, y_train)

    x_input = pd.DataFrame(np.random.rand(1, 5), columns=cols)
    prediction = model.predict(x_input)
    assert len(prediction) == 1


def test_many_samples(model: LightGBMModel) -> None:
    """Many: Test with a representative number of samples."""
    cols = [f"f{i}" for i in range(10)]
    x_train = pd.DataFrame(np.random.rand(100, 10), columns=cols)
    y_train = np.random.randint(0, 2, 100)
    model.train(x_train, y_train)

    x_test = pd.DataFrame(np.random.rand(20, 10), columns=cols)
    predictions = model.predict(x_test)
    assert len(predictions) == 20


def test_boundary_features(model: LightGBMModel) -> None:
    """Boundary: Test with minimum number of features (1)."""
    cols = ["f1"]
    x_train = pd.DataFrame(np.random.rand(10, 1), columns=cols)
    y_train = np.random.randint(0, 2, 10)
    model.train(x_train, y_train)

    x_input = pd.DataFrame(np.random.rand(5, 1), columns=cols)
    predictions = model.predict(x_input)
    assert len(predictions) == 5


def test_exception_invalid_features(model: LightGBMModel) -> None:
    """Exception: Test prediction with mismatching feature count."""
    cols = [f"f{i}" for i in range(5)]
    x_train = pd.DataFrame(np.random.rand(10, 5), columns=cols)
    y_train = np.random.randint(0, 2, 10)
    model.train(x_train, y_train)

    # Mismatching features (different names and count)
    x_invalid = pd.DataFrame(np.random.rand(5, 3), columns=["f1", "f2", "f3"])
    with pytest.raises(Exception):
        model.predict(x_invalid)


def test_simple_happy_path(model: LightGBMModel) -> None:
    """Simple: Test a basic full flow."""
    cols = [f"f{i}" for i in range(5)]
    x_train = pd.DataFrame(np.random.rand(50, 5), columns=cols)
    y_train = np.random.randint(0, 2, 50)
    model.train(x_train, y_train)

    metrics = model.evaluate(x_train, y_train)
    assert "accuracy" in metrics
    for val in metrics.values():
        assert 0.0 <= val <= 1.0
