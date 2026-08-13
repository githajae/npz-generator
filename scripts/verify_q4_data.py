#!/usr/bin/env python3
"""Verify Q4 inputs using the corrected clustered-run implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from npz_generator.loader import load_dataset as load_generated


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-data", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--kernel-project", type=Path, required=True)
    return parser.parse_args()


@jax.jit
def _q4(tables: dict[str, dict[str, jax.Array]]) -> jax.Array:
    orders = tables["orders"]
    lineitem = tables["lineitem"]
    keys = lineitem["l_orderkey"]
    valid = lineitem["__valid_mask__"]
    rank = valid.astype(jnp.int32)
    for distance in range(1, 7):
        previous_keys = jnp.pad(keys[:-distance], (distance, 0))
        previous_valid = jnp.pad(valid[:-distance], (distance, 0))
        rank += (valid & previous_valid & (keys == previous_keys)).astype(jnp.int32)
    values = valid & (lineitem["l_commitdate"] < lineitem["l_receiptdate"])
    accumulated = values | jnp.where(rank > 1, jnp.pad(values[:-1], (1, 0)), False)
    accumulated |= jnp.where(rank > 2, jnp.pad(accumulated[:-2], (2, 0)), False)
    run_has_late = accumulated | jnp.where(
        rank > 4, jnp.pad(accumulated[:-4], (4, 0)), False
    )
    parent = lineitem["__gather_idx_to_orders__"].astype(jnp.int32)
    terminal = valid & jnp.concatenate([parent[:-1] != parent[1:], jnp.array([True])])
    has_late = jnp.zeros(orders["__valid_mask__"].shape[0] + 1, dtype=jnp.int32)
    has_late = (
        has_late.at[jnp.where(terminal, parent, -1)].add(
            jnp.where(terminal, run_has_late, False).astype(jnp.int32)
        )[:-1]
        > 0
    )
    qualifies = (
        orders["__valid_mask__"]
        & has_late
        & (orders["o_orderdate"] >= 19930501)
        & (orders["o_orderdate"] < 19930801)
    )
    priority = orders["o_orderpriority"].astype(jnp.int32)
    return jax.vmap(
        lambda group: jnp.sum((qualifies & (priority == group)).astype(jnp.int32))
    )(jnp.arange(5, dtype=jnp.int32))


def _device(tables: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, object]]:
    staged = jax.tree.map(lambda value: np.array(value, copy=True, order="C"), tables)
    return jax.tree.map(jax.device_put, staged)


def main() -> int:
    arguments = _arguments()
    sys.path.insert(0, str(arguments.kernel_project / "src"))
    from ntdb_tpu.data import load_dataset as load_reference

    projection = {
        "orders": {
            "__valid_mask__",
            "o_orderdate",
            "o_orderpriority",
        },
        "lineitem": {
            "__valid_mask__",
            "__gather_idx_to_orders__",
            "l_orderkey",
            "l_commitdate",
            "l_receiptdate",
        },
    }
    reference = np.asarray(
        jax.device_get(
            _q4(_device(load_reference(arguments.reference_data, projection)))
        )
    )
    generated = np.asarray(
        jax.device_get(
            _q4(_device(load_generated(arguments.generated_data, projection)))
        )
    )
    np.testing.assert_array_equal(generated, reference)
    print(f"Q4: PASS {generated.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
