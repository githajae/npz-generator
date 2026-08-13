"""Independent Parquet-to-NPZ artifact builders."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .config import BuildConfig, normalize_config
from .io import array_metadata, artifact_path, atomic_save_npz, atomic_write_metadata
from .registry import RelationshipSpec, StringSpec, get_workload
from .source import ParquetSource, discover_source

_ASCII_TOKEN = re.compile(r"[A-Za-z]+")


def _source(
    config: BuildConfig, workload: str, sf: int | float, table: str
) -> ParquetSource:
    return discover_source(config.input_root, workload, sf, table, config.snapshot_path)


def _padded_count(row_count: int, multiple: int) -> int:
    return ((row_count + multiple - 1) // multiple) * multiple


def _numeric_dtype(
    data_type: pa.DataType, min_value: int | None, max_value: int | None
) -> np.dtype:
    if pa.types.is_boolean(data_type):
        return np.dtype(np.bool_)
    if pa.types.is_date(data_type) or pa.types.is_timestamp(data_type):
        return np.dtype(np.int32)
    if pa.types.is_floating(data_type) or pa.types.is_decimal(data_type):
        return np.dtype(np.float32)
    if pa.types.is_integer(data_type):
        bounds = np.iinfo(np.int32)
        if (
            min_value is not None
            and min_value >= bounds.min
            and max_value <= bounds.max
        ):
            return np.dtype(np.int32)
        return np.dtype(np.int64)
    raise TypeError(f"unsupported non-string Parquet type {data_type}")


def _column_min_max(
    source: ParquetSource, column: str, batch_size: int
) -> tuple[int | None, int | None]:
    minimum: int | None = None
    maximum: int | None = None
    for batch in source.batches([column], batch_size):
        values = batch.column(0)
        if values.null_count == len(values):
            continue
        batch_min = pc.min(values).as_py()
        batch_max = pc.max(values).as_py()
        minimum = batch_min if minimum is None else min(minimum, batch_min)
        maximum = batch_max if maximum is None else max(maximum, batch_max)
    return minimum, maximum


def _convert_batch(values: pa.Array, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        values.is_valid().to_numpy(zero_copy_only=False).astype(np.bool_, copy=False)
    )
    if np.issubdtype(dtype, np.bool_):
        filled = pc.fill_null(values, False).to_numpy(zero_copy_only=False)
        return np.asarray(filled, dtype=dtype), valid
    if np.issubdtype(dtype, np.integer) and (
        pa.types.is_date(values.type) or pa.types.is_timestamp(values.type)
    ):
        py_values = values.to_pylist()
        converted = np.zeros(len(py_values), dtype=np.int32)
        for index, value in enumerate(py_values):
            if value is not None:
                converted[index] = value.year * 10_000 + value.month * 100 + value.day
        return converted, valid
    fill_value = 0
    filled = pc.fill_null(values, fill_value)
    return np.asarray(filled.to_numpy(zero_copy_only=False), dtype=dtype), valid


def _metadata(
    kind: str,
    workload: str,
    sf: int | float,
    table: str,
    name: str,
    source: ParquetSource,
    arrays: Mapping[str, np.ndarray],
    checksum: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "workload": workload,
        "scale_factor": sf,
        "table": table,
        "name": name,
        "source_rows": source.row_count,
        "snapshot_id": source.snapshot_id,
        "source_fragments": [path.name for path in source.fragments],
        "arrays": array_metadata(arrays),
        "output_sha256": checksum,
        **extra,
    }


def build_column_npz(
    cls: type[BuildConfig] | BuildConfig,
    workload: str,
    sf: int | float,
    tname: str,
    col_name: str,
) -> Path:
    config = normalize_config(cls)
    source = _source(config, workload, sf, tname)
    try:
        field = source.schema.field(col_name)
    except KeyError as error:
        raise ValueError(f"unknown column {tname}.{col_name}") from error
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        raise TypeError(f"{tname}.{col_name} is a string; use build_string_npz")

    minimum, maximum = _column_min_max(source, col_name, config.batch_size)
    dtype = _numeric_dtype(field.type, minimum, maximum)
    row_count = source.row_count
    output_count = _padded_count(row_count, config.padding_multiple)
    with tempfile.TemporaryDirectory(prefix="npzgen-column-") as directory:
        values = np.lib.format.open_memmap(
            Path(directory) / "values.npy",
            mode="w+",
            dtype=dtype,
            shape=(output_count,),
        )
        valid = np.lib.format.open_memmap(
            Path(directory) / "valid.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(output_count,),
        )
        values[:] = 0
        valid[:] = False
        offset = 0
        for batch in source.batches([col_name], config.batch_size):
            converted, batch_valid = _convert_batch(batch.column(0), dtype)
            end = offset + len(converted)
            values[offset:end] = converted
            valid[offset:end] = batch_valid
            offset = end
        arrays: dict[str, np.ndarray] = {col_name: values}
        if field.nullable or output_count != row_count or not bool(np.all(valid)):
            arrays[f"__valid__{col_name}"] = valid
        path = artifact_path(
            config.output_root, workload, sf, tname, "column", col_name
        )
        checksum = atomic_save_npz(path, arrays)
        atomic_write_metadata(
            path,
            _metadata(
                "column",
                workload,
                sf,
                tname,
                col_name,
                source,
                arrays,
                checksum,
                source_type=str(field.type),
                minimum=minimum,
                maximum=maximum,
            ),
        )
    return path


def _iter_strings(
    source: ParquetSource, column: str, batch_size: int
) -> Iterator[str | None]:
    for batch in source.batches([column], batch_size):
        yield from batch.column(0).to_pylist()


def _string_dictionary(
    source: ParquetSource, column: str, spec: StringSpec, batch_size: int
) -> np.ndarray:
    if spec.mode == "row_dictionary":
        return np.asarray(
            [
                "" if value is None else value
                for value in _iter_strings(source, column, batch_size)
            ]
        )
    if spec.mode != "dictionary":
        raise ValueError(f"unsupported string encoding mode {spec.mode!r}")
    unique: set[str] = set()
    for value in _iter_strings(source, column, batch_size):
        if value is not None:
            unique.add(value)
    return np.asarray(sorted(unique))


def _token_arrays(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    tokenized = [_ASCII_TOKEN.findall(value) for value in strings]
    dictionary = np.asarray(sorted({token for row in tokenized for token in row}))
    if len(dictionary) < np.iinfo(np.uint8).max:
        dtype = np.dtype(np.uint8)
    elif len(dictionary) < np.iinfo(np.uint16).max:
        dtype = np.dtype(np.uint16)
    else:
        dtype = np.dtype(np.int32)
    sentinel = np.iinfo(dtype).max if np.issubdtype(dtype, np.unsignedinteger) else -1
    slots = np.full(
        (max((len(row) for row in tokenized), default=0), len(strings)),
        sentinel,
        dtype=dtype,
    )
    ids = {token: index for index, token in enumerate(dictionary)}
    for row_index, tokens in enumerate(tokenized):
        for slot, token in enumerate(tokens):
            slots[slot, row_index] = ids[token]
    return slots, dictionary


def _string_pool(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = [value.encode("utf-8") for value in strings]
    lengths = np.fromiter((len(value) for value in encoded), dtype=np.int64)
    offsets = np.zeros(len(strings), dtype=np.int64)
    if len(strings) > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    total_bytes = int(lengths.sum())
    offset_dtype = np.int32 if total_bytes <= np.iinfo(np.int32).max else np.int64
    return (
        offsets.astype(offset_dtype),
        lengths.astype(offset_dtype),
        np.frombuffer(b"".join(encoded), dtype=np.uint8).copy(),
    )


def build_string_npz(
    cls: type[BuildConfig] | BuildConfig,
    workload: str,
    sf: int | float,
    tname: str,
    col_name: str,
) -> Path:
    config = normalize_config(cls)
    source = _source(config, workload, sf, tname)
    try:
        field = source.schema.field(col_name)
    except KeyError as error:
        raise ValueError(f"unknown column {tname}.{col_name}") from error
    if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
        raise TypeError(f"{tname}.{col_name} is not a string; use build_column_npz")

    workload_spec = get_workload(workload, config.workload_specs)
    spec = workload_spec.string_spec(tname, col_name)
    dictionary = _string_dictionary(source, col_name, spec, config.batch_size)
    row_count = source.row_count
    if row_count > np.iinfo(np.int32).max:
        code_dtype = np.int64
    else:
        code_dtype = np.int32
    if spec.mode == "row_dictionary":
        codes = np.arange(row_count, dtype=code_dtype)
        strings = dictionary
        valid = np.fromiter(
            (
                value is not None
                for value in _iter_strings(source, col_name, config.batch_size)
            ),
            dtype=np.bool_,
            count=row_count,
        )
        codes[~valid] = -1
    else:
        ids = {value: index for index, value in enumerate(dictionary.tolist())}
        codes = np.fromiter(
            (
                -1 if value is None else ids[value]
                for value in _iter_strings(source, col_name, config.batch_size)
            ),
            dtype=code_dtype,
            count=row_count,
        )
        valid = codes >= 0
        strings = np.asarray(
            [
                "" if value is None else value
                for value in _iter_strings(source, col_name, config.batch_size)
            ]
        )
    arrays: dict[str, np.ndarray] = {
        col_name: codes,
        f"__dictionary__{col_name}": dictionary,
    }
    if field.nullable or not bool(np.all(valid)):
        arrays[f"__valid__{col_name}"] = valid
    if spec.tokenize:
        slots, token_dictionary = _token_arrays(strings)
        arrays[f"__token_slots__{col_name}"] = slots
        arrays[f"__dictionary__{col_name}_tokens"] = token_dictionary
    if spec.legacy_pool and config.emit_legacy_string_pools:
        offsets, lengths, pool = _string_pool(strings)
        arrays[f"{col_name}_offset"] = offsets
        arrays[f"{col_name}_length"] = lengths
        arrays[f"{col_name}_pool"] = pool

    path = artifact_path(config.output_root, workload, sf, tname, "string", col_name)
    checksum = atomic_save_npz(path, arrays)
    atomic_write_metadata(
        path,
        _metadata(
            "string",
            workload,
            sf,
            tname,
            col_name,
            source,
            arrays,
            checksum,
            encoding=spec.mode,
            tokenized=spec.tokenize,
            legacy_pool=spec.legacy_pool and config.emit_legacy_string_pools,
        ),
    )
    return path


def _iter_key_tuples(
    source: ParquetSource, columns: Sequence[str], batch_size: int
) -> Iterator[tuple[object, ...] | None]:
    for batch in source.batches(columns, batch_size):
        rows = zip(*(batch.column(index).to_pylist() for index in range(len(columns))))
        for row in rows:
            yield None if any(value is None for value in row) else tuple(row)


def _build_lookup(
    parent: ParquetSource,
    child: ParquetSource,
    relation: RelationshipSpec,
    batch_size: int,
) -> tuple[np.ndarray, int, int]:
    if parent.row_count > np.iinfo(np.int32).max:
        raise OverflowError(
            f"parent {parent.table} has {parent.row_count} rows; "
            "gather positions exceed int32"
        )
    lookup: dict[tuple[object, ...], int] = {}
    for position, key in enumerate(
        _iter_key_tuples(parent, relation.parent_columns, batch_size)
    ):
        if key is None:
            raise ValueError(
                f"primary key {parent.table}.{relation.parent_columns} is NULL"
            )
        if key in lookup:
            raise ValueError(f"duplicate primary key {key!r} in {parent.table}")
        lookup[key] = position
    gather = np.full(child.row_count, -1, dtype=np.int32)
    matched = 0
    nulls = 0
    for position, key in enumerate(
        _iter_key_tuples(child, relation.child_columns, batch_size)
    ):
        if key is None:
            nulls += 1
            continue
        mapped = lookup.get(key, -1)
        gather[position] = mapped
        matched += mapped >= 0
    return gather, matched, nulls


def build_index_npz(
    cls: type[BuildConfig] | BuildConfig,
    workload: str,
    sf: int | float,
    tname: str,
    col_name: str,
) -> Path:
    config = normalize_config(cls)
    relation = get_workload(workload, config.workload_specs).relationship(
        tname, col_name
    )
    child = _source(config, workload, sf, relation.child_table)
    parent = _source(config, workload, sf, relation.parent_table)
    gather, matched, nulls = _build_lookup(parent, child, relation, config.batch_size)
    arrays = {relation.array_name: gather}
    path = artifact_path(config.output_root, workload, sf, tname, "index", col_name)
    checksum = atomic_save_npz(path, arrays)
    atomic_write_metadata(
        path,
        _metadata(
            "index",
            workload,
            sf,
            tname,
            col_name,
            child,
            arrays,
            checksum,
            parent_table=relation.parent_table,
            parent_snapshot_id=parent.snapshot_id,
            child_columns=list(relation.child_columns),
            parent_columns=list(relation.parent_columns),
            matched_rows=matched,
            unmatched_rows=child.row_count - matched - nulls,
            null_rows=nulls,
        ),
    )
    return path


def build_system_npz(
    cls: type[BuildConfig] | BuildConfig,
    workload: str,
    sf: int | float,
    tname: str,
    col_name: str = "__valid_mask__",
) -> Path:
    if col_name != "__valid_mask__":
        raise ValueError("the only system artifact is __valid_mask__")
    config = normalize_config(cls)
    source = _source(config, workload, sf, tname)
    output_count = _padded_count(source.row_count, config.padding_multiple)
    valid = np.zeros(output_count, dtype=np.bool_)
    valid[: source.row_count] = True
    arrays = {"__valid_mask__": valid}
    path = artifact_path(config.output_root, workload, sf, tname, "system", col_name)
    checksum = atomic_save_npz(path, arrays)
    atomic_write_metadata(
        path,
        _metadata(
            "system",
            workload,
            sf,
            tname,
            col_name,
            source,
            arrays,
            checksum,
            padding_rows=output_count - source.row_count,
        ),
    )
    return path
