# Spec: Parquet-to-NPZ Generator

## Objective

Convert canonically ordered, possibly partitioned TPC-H or TPC-DS Parquet
tables into deterministic NPZ artifacts that JAX/TPU query kernels can load
directly. Every builder is an independent container job and reads only the
immutable Parquet snapshot and workload registry.

## Public API

```python
build_column_npz(cls, workload, sf, tname, col_name)
build_string_npz(cls, workload, sf, tname, col_name)
build_index_npz(cls, workload, sf, tname, col_name)
build_system_npz(cls, workload, sf, tname, col_name)
```

`cls` is a `BuildConfig` subclass or instance that supplies input/output roots,
batch size, and optional snapshot path. `col_name` identifies a physical column
for column/string jobs, a registered relationship for index jobs, and must be
`__valid_mask__` for system jobs.

## Artifact Contract

- `column/<column>.npz`: typed numeric/date/boolean array and, for nullable
  columns, `__valid__<column>`.
- `string/<column>.npz`: deterministic lexicographic dictionary codes plus
  `__dictionary__<column>`. Token-mode columns additionally contain fixed-width
  token slots and a token dictionary. Row-dictionary mode preserves payload
  strings without a row-level byte pool.
- `index/<relationship>.npz`: child-to-parent physical row-position gather with
  `-1` for null or unmatched keys. Only registered PK-FK relationships may be
  persisted.
- `system/__valid_mask__.npz`: a table-shaped Boolean validity vector. With no
  requested padding, every physical source row is valid.

Each NPZ has an independently written `.meta.json` sidecar. No builder mutates
a shared manifest. A separate finalizer may collect sidecars after all jobs
finish.

## Source and Ordering

Input fragments are discovered from `<input>/<workload>/sf<sf>/<table>` or from
an immutable snapshot. Natural filename ordering defines physical row order,
so `part-2` precedes `part-10`. The fragment list and file metadata are hashed
into a snapshot identifier stored in every sidecar.

## Commands

```bash
python -m pytest
ruff check .
npzgen build-column --workload tpch --sf 1 --table lineitem --column l_quantity
npzgen build-string --workload tpch --sf 1 --table orders --column o_orderstatus
npzgen build-index --workload tpch --sf 1 --table lineitem --column orders
npzgen build-system --workload tpch --sf 1 --table lineitem
npzgen finalize --output-root ./npz --workload tpch --sf 1
```

## Project Structure

- `src/npz_generator`: registry, builders, Parquet I/O, CLI, finalizer
- `tests`: unit and integration tests
- `scripts`: real-data smoke generation and validation
- `docs`: artifact specification

## Testing Strategy

Unit tests cover type conversion, dictionary ordering, nullable values, natural
fragment ordering, and PK-FK mapping. Integration tests create split Parquet
fragments, run all four independent builders, reload every NPZ, and compare the
decoded data with the Parquet source. A real-data smoke script invokes
`tpchgen-cli parquet --parts=...` and validates generated TPC-H artifacts.

## Boundaries

- Always: stream Parquet in bounded batches; write atomically; validate schema,
  range, and row counts; make output deterministic and idempotent.
- Ask first: change array names consumed by kernels; change physical numeric
  types; add persistent index families.
- Never: persist reverse CSR, start/count, fanout ordinal/rank, repeat-count,
  predicate, or covering indexes; rely on filesystem listing order; commit data
  outputs or credentials.

## Success Criteria

- All four builders work independently against split TPC-H Parquet input.
- The registry represents both TPC-H and TPC-DS schemas and PK-FK roles.
- Round-trip validation proves values, strings, nulls, masks, and gathers.
- Builders are atomic, deterministic, idempotent, and emit auditable sidecars.
- The real `tpchgen-cli` split-Parquet smoke test passes.

