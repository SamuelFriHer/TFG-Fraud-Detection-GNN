"""Unit tests for the CLI helper functions."""

import pytest

from src.cli import _resolve_model_names


def test_resolve_model_names_raises_value_error_on_unknown_model() -> None:
    """Verifies that _resolve_model_names raises ValueError for unrecognized models."""
    invalid_model_names: list[str] = ["invalid_model_name"]
    with pytest.raises(ValueError, match="Unknown model"):
        _resolve_model_names(invalid_model_names)
