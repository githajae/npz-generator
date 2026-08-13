#!/usr/bin/env python3
"""Generate split TPC-H Parquet and validate all NPZ artifact families."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

from npz_generator import (
    BuildConfig,
    build_column_npz,
    build_index_npz,
    build_string_npz,
    build_system_npz,
)
from npz_generator.finalize import finalize_manifest
from npz_generator.registry import TPCH
from npz_generator.source import discover_source


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpchgen-cli", default="tpchgen-cli")
    parser.add_argument("--sf", type=float, default=0.01)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def _decode_string(path: Path, column: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        dictionary = artifact[f"__dictionary__{column}"]
        codes = artifact[column]
        decoded = np.full(len(codes), None, dtype=object)
        valid = codes >= 0
        decoded[valid] = dictionary[codes[valid]]
        return decoded


def main() -> int:
    arguments = _arguments()
    owned = arguments.work_dir is None
    root = arguments.work_dir or Path(tempfile.mkdtemp(prefix="npzgen-tpch-smoke-"))
    parquet_root = root / "parquet"
    output_root = root / "npz"
    tables = "region,nation,customer,orders,lineitem"
    try:
        subprocess.run(
            [
                arguments.tpchgen_cli,
                "parquet",
                "-s",
                str(arguments.sf),
                f"--tables={tables}",
                f"--parts={arguments.parts}",
                f"--output-dir={parquet_root}",
                "--no-progress",
            ],
            check=True,
        )
        config = BuildConfig(
            input_root=parquet_root,
            output_root=output_root,
            batch_size=1_024,
        )
        outputs: list[Path] = []
        for table in tables.split(","):
            source = discover_source(parquet_root, "tpch", arguments.sf, table)
            outputs.append(
                build_system_npz(config, "tpch", arguments.sf, table, "__valid_mask__")
            )
            for field in source.schema:
                if pa.types.is_string(field.type) or pa.types.is_large_string(
                    field.type
                ):
                    output = build_string_npz(
                        config, "tpch", arguments.sf, table, field.name
                    )
                    decoded = _decode_string(output, field.name)
                    original = np.asarray(
                        [
                            value
                            for batch in source.batches([field.name], 1_024)
                            for value in batch.column(0).to_pylist()
                        ],
                        dtype=object,
                    )
                    np.testing.assert_array_equal(decoded, original)
                else:
                    output = build_column_npz(
                        config, "tpch", arguments.sf, table, field.name
                    )
                outputs.append(output)
        for (child_table, role), _relation in TPCH.relationships.items():
            if child_table not in tables.split(","):
                continue
            if _relation.parent_table not in tables.split(","):
                continue
            outputs.append(
                build_index_npz(config, "tpch", arguments.sf, child_table, role)
            )
        manifest = finalize_manifest(output_root, "tpch", arguments.sf)
        print(
            f"PASS: {len(outputs)} deterministic artifacts; manifest={manifest}; "
            f"workspace={root}"
        )
        return 0
    finally:
        if owned and not arguments.keep:
            shutil.rmtree(root)


if __name__ == "__main__":
    raise SystemExit(main())
