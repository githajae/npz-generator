#!/usr/bin/env python3
"""Compare generated arrays with an existing column-sharded SF dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    generated_manifest = json.loads((arguments.generated / "manifest.json").read_text())
    reference_manifest = json.loads((arguments.reference / "manifest.json").read_text())
    reference_files = {
        (table, name): arguments.reference / filename
        for table, details in reference_manifest["tables"].items()
        for name, filename in details.get("column_files", {}).items()
    }
    compared = 0
    generated_only: list[str] = []
    for details in generated_manifest["artifacts"]:
        table = details["table"]
        with np.load(
            arguments.generated / details["path"], allow_pickle=False
        ) as artifact:
            for name in artifact.files:
                reference_path = reference_files.get((table, name))
                if reference_path is None:
                    generated_only.append(f"{table}.{name}")
                    continue
                reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
                np.testing.assert_array_equal(
                    artifact[name],
                    reference,
                    err_msg=f"array mismatch: {table}.{name}",
                )
                compared += 1
    print(
        f"PASS: {compared} arrays exactly match the reference; "
        f"{len(generated_only)} generated-only arrays"
    )
    if generated_only:
        print("generated-only: " + ", ".join(sorted(generated_only)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
