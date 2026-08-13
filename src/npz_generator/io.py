"""Atomic NPZ and metadata output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def format_scale_factor(sf: int | float) -> str:
    value = float(sf)
    return str(int(value)) if value.is_integer() else format(value, "g")


def artifact_path(
    output_root: Path,
    workload: str,
    sf: int | float,
    table: str,
    kind: str,
    name: str,
) -> Path:
    return (
        output_root
        / workload
        / f"sf{format_scale_factor(sf)}"
        / table
        / kind
        / f"{name}.npz"
    )


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            np.savez(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise RuntimeError(f"failed to validate temporary artifact {temporary}")
        checksum = hashlib.sha256()
        with temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        os.replace(temporary, path)
        return checksum.hexdigest()
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_metadata(path: Path, payload: dict[str, object]) -> Path:
    metadata_path = path.with_suffix(".meta.json")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=metadata_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(
                payload,
                destination,
                default=_json_default,
                indent=2,
                sort_keys=True,
            )
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, metadata_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata_path


def array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
