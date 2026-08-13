#!/usr/bin/env python3
"""Sequential convenience driver; production containers should run one builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa

from npz_generator import (
    BuildConfig,
    build_column_npz,
    build_index_npz,
    build_string_npz,
    build_system_npz,
    finalize_manifest,
)
from npz_generator.registry import get_workload
from npz_generator.source import discover_source

TPCH_TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workload", choices=("tpch", "tpcds"), required=True)
    parser.add_argument("--sf", type=float, required=True)
    parser.add_argument("--tables", help="comma-separated; default discovers all")
    parser.add_argument(
        "--columns-json",
        type=Path,
        help="optional {table: [column, ...]} source projection",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    parser.add_argument("--legacy-string-pools", action="store_true")
    return parser.parse_args()


def _tables(arguments: argparse.Namespace) -> tuple[str, ...]:
    if arguments.tables:
        return tuple(arguments.tables.split(","))
    if arguments.workload == "tpch":
        return TPCH_TABLES
    candidates = arguments.input_root / arguments.workload / f"sf{arguments.sf:g}"
    if not candidates.is_dir():
        candidates = arguments.input_root
    return tuple(sorted(path.name for path in candidates.iterdir() if path.is_dir()))


def main() -> int:
    arguments = _arguments()
    tables = _tables(arguments)
    config = BuildConfig(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        batch_size=arguments.batch_size,
        temp_root=arguments.temp_root,
        duckdb_memory_limit=arguments.duckdb_memory_limit,
        emit_legacy_string_pools=arguments.legacy_string_pools,
    )
    artifacts = 0
    column_projection = (
        json.loads(arguments.columns_json.read_text())
        if arguments.columns_json
        else None
    )
    for table in tables:
        source = discover_source(
            config.input_root, arguments.workload, arguments.sf, table
        )
        build_system_npz(
            config, arguments.workload, arguments.sf, table, "__valid_mask__"
        )
        artifacts += 1
        selected = (
            set(column_projection[table]) if column_projection is not None else None
        )
        if selected is not None:
            missing = selected - set(source.schema.names)
            if missing:
                raise ValueError(
                    f"{table} is missing projected columns {sorted(missing)}"
                )
        for field in source.schema:
            if selected is not None and field.name not in selected:
                continue
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                build_string_npz(
                    config, arguments.workload, arguments.sf, table, field.name
                )
            else:
                build_column_npz(
                    config, arguments.workload, arguments.sf, table, field.name
                )
            artifacts += 1
        print(f"{table}: base artifacts complete", flush=True)
    workload = get_workload(arguments.workload)
    for (child, role), relation in workload.relationships.items():
        if child in tables and relation.parent_table in tables:
            build_index_npz(config, arguments.workload, arguments.sf, child, role)
            artifacts += 1
    manifest = finalize_manifest(config.output_root, arguments.workload, arguments.sf)
    print(f"PASS: {artifacts} artifacts; manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
