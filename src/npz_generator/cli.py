"""Command-line boundary for one-artifact-per-container execution."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from .builders import (
    build_column_npz,
    build_index_npz,
    build_string_npz,
    build_system_npz,
)
from .config import BuildConfig
from .finalize import finalize_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npzgen", description="Build TPU-ready NPZ artifacts from Parquet"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build-column", "build-string", "build-index", "build-system"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input-root", type=Path, required=True)
        subparser.add_argument("--output-root", type=Path, required=True)
        subparser.add_argument("--workload", choices=("tpch", "tpcds"), required=True)
        subparser.add_argument("--sf", type=float, required=True)
        subparser.add_argument("--table", required=True)
        if command == "build-system":
            subparser.set_defaults(column="__valid_mask__")
        else:
            help_text = "relationship role" if command == "build-index" else "column"
            subparser.add_argument("--column", required=True, help=help_text)
        subparser.add_argument("--snapshot", type=Path)
        subparser.add_argument("--batch-size", type=int, default=1_000_000)
        subparser.add_argument("--padding-multiple", type=int, default=1)
        subparser.add_argument("--temp-root", type=Path)
        subparser.add_argument("--duckdb-memory-limit", default="8GB")
        if command == "build-string":
            subparser.add_argument("--legacy-string-pools", action="store_true")

    finalizer = subparsers.add_parser("finalize")
    finalizer.add_argument("--output-root", type=Path, required=True)
    finalizer.add_argument("--workload", choices=("tpch", "tpcds"), required=True)
    finalizer.add_argument("--sf", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "finalize":
        print(
            finalize_manifest(arguments.output_root, arguments.workload, arguments.sf)
        )
        return 0
    config = BuildConfig(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        snapshot_path=arguments.snapshot,
        batch_size=arguments.batch_size,
        padding_multiple=arguments.padding_multiple,
        temp_root=arguments.temp_root,
        duckdb_memory_limit=arguments.duckdb_memory_limit,
        emit_legacy_string_pools=getattr(arguments, "legacy_string_pools", False),
    )
    builders: dict[str, Callable[..., Path]] = {
        "build-column": build_column_npz,
        "build-string": build_string_npz,
        "build-index": build_index_npz,
        "build-system": build_system_npz,
    }
    output = builders[arguments.command](
        config,
        arguments.workload,
        arguments.sf,
        arguments.table,
        arguments.column,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
