"""Deterministic discovery and bounded-batch access to Parquet fragments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_NATURAL_NUMBER = re.compile(r"(\d+)")


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_NUMBER.split(path.as_posix())
    )


@dataclass(frozen=True)
class ParquetSource:
    table: str
    fragments: tuple[Path, ...]
    snapshot_id: str

    @property
    def row_count(self) -> int:
        return sum(
            pq.ParquetFile(fragment).metadata.num_rows for fragment in self.fragments
        )

    @property
    def schema(self) -> pa.Schema:
        return pq.ParquetFile(self.fragments[0]).schema_arrow

    def batches(
        self, columns: Sequence[str], batch_size: int
    ) -> Iterator[pa.RecordBatch]:
        expected = set(columns)
        for fragment in self.fragments:
            parquet = pq.ParquetFile(fragment)
            missing = expected - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{fragment} is missing columns {sorted(missing)}")
            yield from parquet.iter_batches(
                columns=list(columns), batch_size=batch_size
            )


def _source_candidates(
    root: Path, workload: str, sf: int | float, table: str
) -> list[Path]:
    sf_text = str(sf).rstrip("0").rstrip(".") if isinstance(sf, float) else str(sf)
    return [
        root / workload / f"sf{sf_text}" / table,
        root / f"sf{sf_text}" / table,
        root / table,
        root / workload / f"sf{sf_text}",
        root / f"sf{sf_text}",
        root,
    ]


def discover_source(
    input_root: Path,
    workload: str,
    sf: int | float,
    table: str,
    snapshot_path: Path | None = None,
) -> ParquetSource:
    root = input_root.resolve()
    if snapshot_path:
        payload = json.loads(snapshot_path.read_text())
        entries = payload["tables"][table]
        fragments = tuple((root / entry).resolve() for entry in entries)
    else:
        fragments = ()
        for candidate in _source_candidates(root, workload, sf, table):
            if candidate.is_dir():
                found = sorted(candidate.glob("*.parquet"), key=natural_key)
            elif candidate.suffix == ".parquet" and candidate.is_file():
                found = [candidate]
            else:
                found = []
            if found:
                fragments = tuple(path.resolve() for path in found)
                break
    if not fragments:
        raise FileNotFoundError(
            f"no Parquet fragments found for {workload}/sf{sf}/{table} under {root}"
        )
    schemas = [pq.ParquetFile(path).schema_arrow for path in fragments]
    if any(
        not schemas[0].equals(schema, check_metadata=False) for schema in schemas[1:]
    ):
        raise ValueError(f"Parquet fragments for {table} have inconsistent schemas")

    digest = hashlib.sha256()
    for fragment in fragments:
        try:
            relative = fragment.relative_to(root)
        except ValueError:
            relative = fragment
        metadata = pq.ParquetFile(fragment).metadata
        digest.update(relative.as_posix().encode())
        digest.update(str(fragment.stat().st_size).encode())
        digest.update(str(metadata.num_rows).encode())
        digest.update(str(metadata.serialized_size).encode())
    return ParquetSource(table, fragments, digest.hexdigest())
