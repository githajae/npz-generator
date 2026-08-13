# NPZ Generator

`npz-generator` converts partitioned TPC-H and TPC-DS Parquet tables into
deterministic, TPU-ready NPZ artifacts. Each artifact is built by an independent
job, so column, string, PK-FK index, and validity-mask generation can run in
separate containers without coordinating writes.

## Artifact families

| Builder | Output | Persistent contents |
|---|---|---|
| `build_column_npz` | `column/<column>.npz` | Numeric/date/Boolean values and NULL validity |
| `build_string_npz` | `string/<column>.npz` | Dictionary codes, dictionary, optional token slots |
| `build_index_npz` | `index/<role>.npz` | Registered PK-FK child-to-parent gather only |
| `build_system_npz` | `system/__valid_mask__.npz` | Physical-row validity/padding mask |

Every NPZ has a `.meta.json` sidecar with its source snapshot, array contract,
row count, and checksum. Builders never modify a shared manifest. Run `finalize`
once after distributed jobs finish.

## Install and test

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check .
```

Generate real split TPC-H Parquet with `tpchgen-cli` and validate all four
artifact families:

```bash
.venv/bin/pip install tpchgen-cli==3.0.0
.venv/bin/python scripts/smoke_tpch.py \
  --tpchgen-cli .venv/bin/tpchgen-cli --sf 0.01 --parts 2
```

## One job per container

The input root may be the `tpchgen-cli` output directory itself, or a dataset
root such as `<root>/tpch/sf1000/<table>/*.parquet`.

```bash
npzgen build-column \
  --input-root /data/parquet --output-root /data/npz \
  --workload tpch --sf 100 --table lineitem --column l_quantity

npzgen build-string \
  --input-root /data/parquet --output-root /data/npz \
  --workload tpch --sf 100 --table orders --column o_comment

npzgen build-index \
  --input-root /data/parquet --output-root /data/npz \
  --workload tpch --sf 100 --table lineitem --column orders

npzgen build-system \
  --input-root /data/parquet --output-root /data/npz \
  --workload tpch --sf 100 --table lineitem

npzgen finalize --output-root /data/npz --workload tpch --sf 100
```

String pools are disabled by default. Dictionary codes support equality and
prefix predicates, and registered token columns provide word-level matching.
Use `--legacy-string-pools` only for kernels that still require byte-offset
substring scans.

## Persistent-index policy

Only registered child-to-parent PK-FK gathers are emitted. Reverse CSR,
start/count, fanout ordinals, repeat counts, predicate indexes, and covering
indexes are intentionally not generated; kernels must derive them on the fly.

See [docs/spec.md](docs/spec.md) for the full contract.

