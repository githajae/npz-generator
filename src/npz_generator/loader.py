"""Load finalized artifacts into the table dictionary expected by JAX kernels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_dataset(
    path: Path, projection: dict[str, set[str]] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads((path / "manifest.json").read_text())
    tables: dict[str, dict[str, np.ndarray]] = {}
    for details in manifest["artifacts"]:
        table = details["table"]
        if projection is not None and table not in projection:
            continue
        requested = None if projection is None else projection[table]
        available = set(details["arrays"])
        selected = available if requested is None else available & requested
        selected = {name for name in selected if not name.startswith("__dictionary__")}
        if not selected:
            continue
        with np.load(path / details["path"], allow_pickle=False) as artifact:
            target = tables.setdefault(table, {})
            for name in selected:
                if name in target:
                    raise ValueError(f"duplicate array {table}.{name} in manifest")
                target[name] = artifact[name]
    if projection is not None:
        for table, requested in projection.items():
            found = set(tables.get(table, {}))
            missing = {
                name
                for name in requested - found
                if not name.startswith("__dictionary__")
            }
            if missing:
                raise KeyError(f"{table} projection is missing {sorted(missing)}")
    return tables
