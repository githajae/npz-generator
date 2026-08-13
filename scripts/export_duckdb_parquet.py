#!/usr/bin/env python3
"""Export a DuckDB database to naturally ordered Parquet fragments."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.parts <= 0:
        raise ValueError("parts must be positive")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(arguments.database), read_only=True)
    tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
    for table in tables:
        quoted_table = _quote_identifier(table)
        rows = connection.execute(f"SELECT count(*) FROM {quoted_table}").fetchone()[0]
        width = max(1, (rows + arguments.parts - 1) // arguments.parts)
        table_root = arguments.output_root / table
        table_root.mkdir(parents=True, exist_ok=True)
        for part in range(arguments.parts):
            start = part * width
            end = min(rows, start + width)
            if start >= end:
                break
            destination = table_root / f"{table}.{part + 1}.parquet"
            connection.execute(
                "COPY ("
                f"SELECT * FROM {quoted_table} "
                f"WHERE rowid >= {start} AND rowid < {end} ORDER BY rowid"
                f") TO {_quote_literal(str(destination))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        print(f"{table}: {rows} rows", flush=True)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
