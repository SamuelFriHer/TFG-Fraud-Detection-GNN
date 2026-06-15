"""Unit tests for the ExperimentConfig parser and validator."""

import tomllib
from pathlib import Path

import pytest

from src.config.experiment_config import ExperimentConfig


def test_from_toml_with_valid_config(tmp_path: Path) -> None:
    """Verifies parsing of a fully defined and valid TOML config file."""
    config_file_path: Path = tmp_path.joinpath("valid_config.toml")
    config_content: str = """
    [dataset]
    handle = "test/dataset-handle"
    prefix = "test_prefix_"

    [split]
    test_size = 0.25
    random_state = 100

    [models.MEGA_PNA]
    in_channels = 8
    hidden_channels = 64
    """
    config_file_path.write_text(config_content, encoding="utf-8")

    parsed_config: ExperimentConfig = ExperimentConfig.from_toml(str(config_file_path))

    assert parsed_config.dataset.handle == "test/dataset-handle"
    assert parsed_config.dataset.prefix == "test_prefix_"
    assert parsed_config.split.test_size == 0.25
    assert parsed_config.split.random_state == 100
    assert parsed_config.models["MEGA_PNA"]["in_channels"] == 8
    assert parsed_config.models["MEGA_PNA"]["hidden_channels"] == 64


def test_from_toml_with_missing_optional_sections(tmp_path: Path) -> None:
    """Verifies that optional sections get default values when omitted."""
    config_file_path: Path = tmp_path.joinpath("minimal_config.toml")
    config_content: str = """
    [dataset]
    handle = "test/dataset-handle"
    prefix = "test_prefix_"
    """
    config_file_path.write_text(config_content, encoding="utf-8")

    parsed_config: ExperimentConfig = ExperimentConfig.from_toml(str(config_file_path))

    assert parsed_config.dataset.handle == "test/dataset-handle"
    assert parsed_config.dataset.prefix == "test_prefix_"
    # Default values from SplitConfig
    assert parsed_config.split.test_size == 0.4
    assert parsed_config.split.random_state == 42
    # Default factory from ExperimentConfig
    assert parsed_config.models == {}


def test_from_toml_raises_file_not_found_error() -> None:
    """Verifies that from_toml raises FileNotFoundError for non-existent paths."""
    missing_config_path: str = "non_existent_config_file.toml"
    with pytest.raises(FileNotFoundError, match="Config not found"):
        ExperimentConfig.from_toml(missing_config_path)


def test_from_toml_raises_toml_decode_error_for_invalid_syntax(tmp_path: Path) -> None:
    """Verifies that from_toml raises TOMLDecodeError for malformed TOML syntax."""
    config_file_path: Path = tmp_path.joinpath("invalid_syntax.toml")
    malformed_content: str = '[dataset]\nhandle = "test/dataset-handle"\nprefix =\n'
    config_file_path.write_text(malformed_content, encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        ExperimentConfig.from_toml(str(config_file_path))


def test_from_toml_raises_key_error_for_missing_required_section(tmp_path: Path) -> None:
    """Verifies that from_toml raises KeyError if the dataset section is missing."""
    config_file_path: Path = tmp_path.joinpath("missing_dataset.toml")
    config_content: str = """
    [split]
    test_size = 0.25
    """
    config_file_path.write_text(config_content, encoding="utf-8")

    with pytest.raises(KeyError):
        ExperimentConfig.from_toml(str(config_file_path))


def test_from_toml_raises_type_error_for_missing_required_dataset_fields(tmp_path: Path) -> None:
    """Verifies that from_toml raises TypeError if required fields in dataset are missing."""
    config_file_path: Path = tmp_path.joinpath("missing_dataset_fields.toml")
    config_content: str = """
    [dataset]
    handle = "test/dataset-handle"
    """
    config_file_path.write_text(config_content, encoding="utf-8")

    with pytest.raises(TypeError, match="missing.*argument"):
        ExperimentConfig.from_toml(str(config_file_path))


def test_from_toml_raises_type_error_for_invalid_extra_keys(tmp_path: Path) -> None:
    """Verifies that from_toml raises TypeError if unexpected keys are present."""
    config_file_path: Path = tmp_path.joinpath("extra_keys.toml")
    config_content: str = """
    [dataset]
    handle = "test/dataset-handle"
    prefix = "test_prefix_"
    unexpected_field = "value"
    """
    config_file_path.write_text(config_content, encoding="utf-8")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ExperimentConfig.from_toml(str(config_file_path))
