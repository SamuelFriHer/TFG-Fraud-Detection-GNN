"""Typed experiment configuration loaded from TOML files."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Typed wrapper for the [dataset] TOML section."""

    handle: str
    prefix: str


@dataclass(frozen=True)
class SplitConfig:
    """Typed wrapper for the [split] TOML section."""

    test_size: float = 0.4
    random_state: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed wrapper for the full TOML experiment configuration."""

    dataset: DatasetConfig
    split: SplitConfig
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_toml(config_path: str) -> "ExperimentConfig":
        """Parses and validates a TOML config file into a typed object."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "rb") as config_file:
            raw = tomllib.load(config_file)
        return ExperimentConfig(
            dataset=DatasetConfig(**raw["dataset"]),
            split=SplitConfig(**raw.get("split", {})),
            models=raw.get("models", {}),
        )
