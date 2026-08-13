#!/usr/bin/env python3
"""Run JAX kernels on reference and generated inputs and compare packed results."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import jax
import numpy as np

from npz_generator.loader import load_dataset as load_generated


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-data", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--kernel-project", type=Path, required=True)
    parser.add_argument("--queries", default="1-22")
    return parser.parse_args()


def _query_numbers(specification: str) -> list[int]:
    result: set[int] = set()
    for component in specification.split(","):
        if "-" in component:
            start, end = map(int, component.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(component))
    return sorted(result)


def _synchronize(value: object) -> None:
    for leaf in jax.tree.leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _packed(module: object, tables: dict[str, dict[str, object]]) -> np.ndarray:
    result = module.execute_query(tables)
    packed = result["__packed_2d__"]
    overflow = result.get("__shape_overflow__")
    if overflow is None:
        return np.asarray(jax.device_get(packed))
    host_packed, host_overflow = jax.device_get((packed, overflow))
    if bool(np.asarray(host_overflow)):
        raise RuntimeError("compiled shape bound overflow")
    return np.asarray(host_packed)


def main() -> int:
    arguments = _arguments()
    source = arguments.kernel_project / "src"
    generated = arguments.kernel_project / "generated"
    sys.path[:0] = [str(source), str(generated)]
    from ntdb_tpu.data import load_dataset as load_reference
    from ntdb_tpu.projection import queries_projection

    queries = _query_numbers(arguments.queries)
    for query in queries:
        projection = queries_projection(generated, [query])
        reference_host = load_reference(arguments.reference_data, projection)
        generated_host = load_generated(arguments.generated_data, projection)
        reference_device = jax.tree.map(jax.device_put, reference_host)
        generated_device = jax.tree.map(jax.device_put, generated_host)
        module = importlib.import_module(f"ntdb.queries.q{query}")
        reference_output = _packed(module, reference_device)
        generated_output = _packed(module, generated_device)
        _synchronize((reference_output, generated_output))
        np.testing.assert_array_equal(
            generated_output,
            reference_output,
            err_msg=f"Q{query} output differs",
        )
        print(f"Q{query}: PASS {generated_output.shape}", flush=True)
    print(f"PASS: {len(queries)} queries match on {jax.devices()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
