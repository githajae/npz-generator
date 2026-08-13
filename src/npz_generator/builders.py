"""Independent Parquet-to-NPZ artifact builders."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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
        if output_count != row_count or not bool(np.all(valid)):
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


def _profile_strings(
    source: ParquetSource,
    column: str,
    batch_size: int,
    tokenize: bool,
    include_pool: bool,
) -> tuple[int, int, int, set[str]]:
    max_characters = 0
    total_bytes = 0
    max_tokens = 0
    tokens: set[str] = set()
    for value in _iter_strings(source, column, batch_size):
        if value is None:
            continue
        max_characters = max(max_characters, len(value))
        if include_pool:
            total_bytes += len(value.encode("utf-8"))
        if tokenize:
            row_tokens = _ASCII_TOKEN.findall(value)
            max_tokens = max(max_tokens, len(row_tokens))
            tokens.update(row_tokens)
    return max_characters, total_bytes, max_tokens, tokens


def _open_memmap(
    directory: Path, name: str, dtype: np.dtype | type, shape: tuple[int, ...]
) -> np.memmap:
    return np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )


def _build_sorted_dictionary(
    source: ParquetSource,
    column: str,
    directory: Path,
    batch_size: int,
    memory_limit: str,
    max_characters: int,
) -> np.memmap:
    database = directory / "dictionary.duckdb"
    spill = directory / "dictionary-spill"
    spill.mkdir()
    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit={_quote_literal(memory_limit)}")
    connection.execute(f"SET temp_directory={_quote_literal(str(spill))}")
    files = ", ".join(_quote_literal(str(path)) for path in source.fragments)
    identifier = _quote_identifier(column)
    connection.execute(
        "CREATE TABLE dictionary AS "
        f"SELECT DISTINCT {identifier} AS value FROM read_parquet([{files}]) "
        f"WHERE {identifier} IS NOT NULL ORDER BY value"
    )
    count = connection.execute("SELECT count(*) FROM dictionary").fetchone()[0]
    dictionary = _open_memmap(
        directory,
        "dictionary",
        np.dtype(f"<U{max(1, max_characters)}"),
        (count,),
    )
    offset = 0
    reader = connection.execute(
        "SELECT value FROM dictionary ORDER BY value"
    ).fetch_record_batch(rows_per_batch=batch_size)
    for batch in reader:
        values = batch.column(0).to_pylist()
        dictionary[offset : offset + len(values)] = values
        offset += len(values)
    connection.close()
    return dictionary


def _string_build_arrays(
    source: ParquetSource,
    column: str,
    spec: StringSpec,
    config: BuildConfig,
    directory: Path,
) -> dict[str, np.ndarray]:
    include_pool = spec.legacy_pool and config.emit_legacy_string_pools
    max_chars, total_bytes, max_tokens, tokens = _profile_strings(
        source,
        column,
        config.batch_size,
        spec.tokenize,
        include_pool,
    )
    row_count = source.row_count
    code_dtype = np.int64 if row_count > np.iinfo(np.int32).max else np.int32
    if spec.mode == "dictionary":
        dictionary = _build_sorted_dictionary(
            source,
            column,
            directory,
            config.batch_size,
            config.duckdb_memory_limit,
            max_chars,
        )
    elif spec.mode == "row_dictionary":
        dictionary = _open_memmap(
            directory,
            "dictionary",
            np.dtype(f"<U{max(1, max_chars)}"),
            (row_count,),
        )
    else:
        raise ValueError(f"unsupported string encoding mode {spec.mode!r}")

    codes = _open_memmap(directory, "codes", code_dtype, (row_count,))
    valid = _open_memmap(directory, "valid", np.bool_, (row_count,))
    arrays: dict[str, np.ndarray] = {
        column: codes,
        f"__dictionary__{column}": dictionary,
    }
    token_dictionary = np.asarray(sorted(tokens))
    token_ids = {token: index for index, token in enumerate(token_dictionary)}
    if len(token_dictionary) < np.iinfo(np.uint8).max:
        token_dtype = np.dtype(np.uint8)
    elif len(token_dictionary) < np.iinfo(np.uint16).max:
        token_dtype = np.dtype(np.uint16)
    else:
        token_dtype = np.dtype(np.int32)
    if spec.tokenize:
        sentinel = (
            np.iinfo(token_dtype).max
            if np.issubdtype(token_dtype, np.unsignedinteger)
            else -1
        )
        slots = _open_memmap(
            directory, "token-slots", token_dtype, (max_tokens, row_count)
        )
        slots[:] = sentinel
        arrays[f"__token_slots__{column}"] = slots
        arrays[f"__dictionary__{column}_tokens"] = token_dictionary
    if include_pool:
        offset_dtype = np.int32 if total_bytes <= np.iinfo(np.int32).max else np.int64
        offsets = _open_memmap(directory, "offsets", offset_dtype, (row_count,))
        lengths = _open_memmap(directory, "lengths", offset_dtype, (row_count,))
        pool = _open_memmap(directory, "pool", np.uint8, (total_bytes,))
        arrays[f"{column}_offset"] = offsets
        arrays[f"{column}_length"] = lengths
        arrays[f"{column}_pool"] = pool

    row_offset = 0
    byte_offset = 0
    for batch in source.batches([column], config.batch_size):
        raw_values = batch.column(0).to_pylist()
        normalized = ["" if value is None else value for value in raw_values]
        size = len(normalized)
        end = row_offset + size
        batch_valid = np.fromiter(
            (value is not None for value in raw_values), dtype=np.bool_, count=size
        )
        valid[row_offset:end] = batch_valid
        if spec.mode == "row_dictionary":
            dictionary[row_offset:end] = normalized
            codes[row_offset:end] = np.arange(row_offset, end, dtype=code_dtype)
        elif len(dictionary):
            positions = np.searchsorted(dictionary, normalized).astype(
                code_dtype, copy=False
            )
            codes[row_offset:end] = positions
        else:
            codes[row_offset:end] = -1
        codes[row_offset:end][~batch_valid] = -1
        if spec.tokenize:
            for local_row, value in enumerate(normalized):
                for slot, token in enumerate(_ASCII_TOKEN.findall(value)):
                    slots[slot, row_offset + local_row] = token_ids[token]
        if include_pool:
            encoded = [value.encode("utf-8") for value in normalized]
            batch_lengths = np.fromiter(
                (len(value) for value in encoded), dtype=offset_dtype, count=size
            )
            lengths[row_offset:end] = batch_lengths
            offsets[row_offset:end] = byte_offset + np.concatenate(
                (
                    np.zeros(1, dtype=offset_dtype),
                    np.cumsum(batch_lengths[:-1], dtype=offset_dtype),
                )
            )
            packed = b"".join(encoded)
            pool[byte_offset : byte_offset + len(packed)] = np.frombuffer(
                packed, dtype=np.uint8
            )
            byte_offset += len(packed)
        row_offset = end
    if not bool(np.all(valid)):
        arrays[f"__valid__{column}"] = valid
    return arrays


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

    spec = get_workload(workload, config.workload_specs).string_spec(tname, col_name)
    if config.temp_root is not None:
        config.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="npzgen-string-", dir=config.temp_root
    ) as temporary:
        arrays = _string_build_arrays(source, col_name, spec, config, Path(temporary))
        path = artifact_path(
            config.output_root, workload, sf, tname, "string", col_name
        )
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


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _dense_lookup_span(
    parent: ParquetSource,
    relation: RelationshipSpec,
    batch_size: int,
) -> int | None:
    if len(relation.parent_columns) != 1:
        return None
    field = parent.schema.field(relation.parent_columns[0])
    if not pa.types.is_integer(field.type):
        return None
    minimum, maximum = _column_min_max(parent, relation.parent_columns[0], batch_size)
    if minimum is None or maximum is None or minimum < 0:
        return None
    return int(maximum) + 1


def _build_dense_lookup(
    parent: ParquetSource,
    child: ParquetSource,
    relation: RelationshipSpec,
    batch_size: int,
    directory: Path,
    span: int,
) -> tuple[np.memmap, int, int]:
    parent_column = relation.parent_columns[0]
    child_column = relation.child_columns[0]
    lookup = np.lib.format.open_memmap(
        directory / "dense-lookup.npy",
        mode="w+",
        dtype=np.int32,
        shape=(span,),
    )
    lookup[:] = -1
    parent_position = 0
    for batch in parent.batches([parent_column], batch_size):
        arrow_keys = batch.column(0)
        if arrow_keys.null_count:
            raise ValueError(f"primary key {parent.table}.{parent_column} is NULL")
        keys = np.asarray(arrow_keys.to_numpy(zero_copy_only=False), dtype=np.int64)
        existing = lookup[keys]
        if bool(np.any(existing != -1)):
            duplicate = int(keys[np.flatnonzero(existing != -1)[0]])
            raise ValueError(f"duplicate primary key {duplicate!r} in {parent.table}")
        lookup[keys] = np.arange(
            parent_position, parent_position + len(keys), dtype=np.int32
        )
        parent_position += len(keys)

    gather = np.lib.format.open_memmap(
        directory / "gather.npy",
        mode="w+",
        dtype=np.int32,
        shape=(child.row_count,),
    )
    gather[:] = -1
    child_position = 0
    matched = 0
    nulls = 0
    for batch in child.batches([child_column], batch_size):
        arrow_keys = batch.column(0)
        valid = arrow_keys.is_valid().to_numpy(zero_copy_only=False)
        keys = np.asarray(
            pc.fill_null(arrow_keys, 0).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        in_range = valid & (keys >= 0) & (keys < span)
        mapped = np.full(len(keys), -1, dtype=np.int32)
        mapped[in_range] = lookup[keys[in_range]]
        end = child_position + len(keys)
        gather[child_position:end] = mapped
        matched += int(np.count_nonzero(mapped >= 0))
        nulls += int(np.count_nonzero(~valid))
        child_position = end
    return gather, matched, nulls


def _create_external_parent(
    connection: duckdb.DuckDBPyConnection,
    parent: ParquetSource,
    relation: RelationshipSpec,
) -> list[str]:
    aliases = [f"k{index}" for index in range(len(relation.parent_columns))]
    offset = 0
    for fragment_index, fragment in enumerate(parent.fragments):
        keys = ", ".join(
            f"{_quote_identifier(column)} AS {_quote_identifier(alias)}"
            for column, alias in zip(relation.parent_columns, aliases)
        )
        query = (
            f"SELECT {keys}, "
            f"CAST(row_number() OVER () - 1 + {offset} AS BIGINT) AS position "
            f"FROM read_parquet({_quote_literal(str(fragment))})"
        )
        prefix = (
            "CREATE TABLE parent_keys AS"
            if fragment_index == 0
            else "INSERT INTO parent_keys"
        )
        connection.execute(f"{prefix} {query}")
        offset += pq.ParquetFile(fragment).metadata.num_rows
    return aliases


def _build_external_lookup(
    parent: ParquetSource,
    child: ParquetSource,
    relation: RelationshipSpec,
    batch_size: int,
    directory: Path,
    memory_limit: str,
) -> tuple[np.memmap, int, int]:
    database = directory / "lookup.duckdb"
    spill = directory / "spill"
    spill.mkdir()
    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit={_quote_literal(memory_limit)}")
    connection.execute(f"SET temp_directory={_quote_literal(str(spill))}")
    aliases = _create_external_parent(connection, parent, relation)
    null_predicate = " OR ".join(
        f"{_quote_identifier(alias)} IS NULL" for alias in aliases
    )
    if connection.execute(
        f"SELECT count(*) FROM parent_keys WHERE {null_predicate}"
    ).fetchone()[0]:
        raise ValueError(
            f"primary key {parent.table}.{relation.parent_columns} is NULL"
        )
    group_columns = ", ".join(_quote_identifier(alias) for alias in aliases)
    duplicate = connection.execute(
        "SELECT 1 FROM parent_keys "
        f"GROUP BY {group_columns} HAVING count(*) > 1 LIMIT 1"
    ).fetchone()
    if duplicate:
        raise ValueError(f"duplicate primary key in {parent.table}")

    gather = np.lib.format.open_memmap(
        directory / "gather.npy",
        mode="w+",
        dtype=np.int32,
        shape=(child.row_count,),
    )
    gather[:] = -1
    output_offset = 0
    matched = 0
    nulls = 0
    for fragment in child.fragments:
        selections = ", ".join(
            f"{_quote_identifier(column)} AS {_quote_identifier(alias)}"
            for column, alias in zip(relation.child_columns, aliases)
        )
        join = " AND ".join(
            f"child.{_quote_identifier(alias)} = parent_keys.{_quote_identifier(alias)}"
            for alias in aliases
        )
        child_null = " OR ".join(
            f"child.{_quote_identifier(alias)} IS NULL" for alias in aliases
        )
        query = (
            "SELECT COALESCE(parent_keys.position, -1) AS position, "
            f"({child_null}) AS is_null FROM "
            f"(SELECT row_number() OVER () AS source_row, {selections} "
            f"FROM read_parquet({_quote_literal(str(fragment))})) child "
            f"LEFT JOIN parent_keys ON {join} ORDER BY child.source_row"
        )
        reader = connection.execute(query).fetch_record_batch(rows_per_batch=batch_size)
        for batch in reader:
            positions = np.asarray(
                batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int32
            )
            is_null = np.asarray(
                batch.column(1).to_numpy(zero_copy_only=False), dtype=np.bool_
            )
            end = output_offset + len(positions)
            gather[output_offset:end] = positions
            matched += int(np.count_nonzero(positions >= 0))
            nulls += int(np.count_nonzero(is_null))
            output_offset = end
    connection.close()
    return gather, matched, nulls


def _build_lookup(
    parent: ParquetSource,
    child: ParquetSource,
    relation: RelationshipSpec,
    config: BuildConfig,
    directory: Path,
) -> tuple[np.memmap, int, int, str]:
    if parent.row_count > np.iinfo(np.int32).max:
        raise OverflowError(
            f"parent {parent.table} has {parent.row_count} rows; "
            "gather positions exceed int32"
        )
    span = _dense_lookup_span(parent, relation, config.batch_size)
    if span is not None and span <= max(
        parent.row_count * config.dense_index_max_span_ratio,
        parent.row_count + 1_000_000,
    ):
        gather, matched, nulls = _build_dense_lookup(
            parent,
            child,
            relation,
            config.batch_size,
            directory,
            span,
        )
        return gather, matched, nulls, "dense_integer"
    gather, matched, nulls = _build_external_lookup(
        parent,
        child,
        relation,
        config.batch_size,
        directory,
        config.duckdb_memory_limit,
    )
    return gather, matched, nulls, "external_join"


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
    if config.temp_root is not None:
        config.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="npzgen-index-", dir=config.temp_root
    ) as temporary:
        gather, matched, nulls, strategy = _build_lookup(
            parent, child, relation, config, Path(temporary)
        )
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
                lookup_strategy=strategy,
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
