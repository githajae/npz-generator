"""Public configuration boundary for independent artifact jobs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .registry import WorkloadSpec


class BuildConfig:
    input_root: Path = Path("parquet")
    output_root: Path = Path("npz")
    snapshot_path: Path | None = None
    batch_size: int = 1_000_000
    padding_multiple: int = 1
    emit_legacy_string_pools: bool = False
    workload_specs: Mapping[str, WorkloadSpec] = {}

    def __init__(self, **overrides: object) -> None:
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise TypeError(f"unknown BuildConfig option {name!r}")
            setattr(self, name, value)

        self.input_root = Path(self.input_root)
        self.output_root = Path(self.output_root)
        if self.snapshot_path is not None:
            self.snapshot_path = Path(self.snapshot_path)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.padding_multiple <= 0:
            raise ValueError("padding_multiple must be positive")


def normalize_config(config: type[BuildConfig] | BuildConfig) -> BuildConfig:
    return config() if isinstance(config, type) else config
