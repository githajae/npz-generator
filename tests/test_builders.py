import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from npz_generator import (
    BuildConfig,
    build_column_npz,
    build_index_npz,
    build_string_npz,
    build_system_npz,
)


class BuilderConfig(BuildConfig):
    batch_size = 2


def _write_parts(root: Path, table: str, parts: list[pa.Table]) -> None:
    table_dir = root / "tpch" / "sf1" / table
    table_dir.mkdir(parents=True)
    for part_number, part in enumerate(parts, 1):
        pq.write_table(part, table_dir / f"{table}.{part_number}.parquet")


def _config(tmp_path: Path) -> BuilderConfig:
    BuilderConfig.input_root = tmp_path / "parquet"
    BuilderConfig.output_root = tmp_path / "npz"
    return BuilderConfig()


def test_builds_numeric_nullable_and_system_arrays_from_natural_part_order(tmp_path):
    config = _config(tmp_path)
    _write_parts(
        config.input_root,
        "lineitem",
        [
            pa.table({"l_orderkey": [1, 1], "l_quantity": [10.0, None]}),
            pa.table({"l_orderkey": [2], "l_quantity": [30.0]}),
        ],
    )

    column_path = build_column_npz(config, "tpch", 1, "lineitem", "l_quantity")
    system_path = build_system_npz(config, "tpch", 1, "lineitem", "__valid_mask__")

    with np.load(column_path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(
            artifact["l_quantity"], np.array([10, 0, 30], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            artifact["__valid__l_quantity"], [True, False, True]
        )
    with np.load(system_path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact["__valid_mask__"], [True] * 3)


def test_dictionary_string_encoding_is_lexicographic_and_round_trips(tmp_path):
    config = _config(tmp_path)
    _write_parts(
        config.input_root,
        "orders",
        [pa.table({"o_orderstatus": ["O", "F"]}), pa.table({"o_orderstatus": ["P"]})],
    )

    output = build_string_npz(config, "tpch", 1, "orders", "o_orderstatus")

    with np.load(output, allow_pickle=False) as artifact:
        dictionary = artifact["__dictionary__o_orderstatus"]
        codes = artifact["o_orderstatus"]
        np.testing.assert_array_equal(dictionary, ["F", "O", "P"])
        np.testing.assert_array_equal(dictionary[codes], ["O", "F", "P"])


def test_pk_fk_index_maps_child_rows_to_parent_physical_positions(tmp_path):
    config = _config(tmp_path)
    _write_parts(
        config.input_root,
        "orders",
        [pa.table({"o_orderkey": [10, 30]}), pa.table({"o_orderkey": [20]})],
    )
    _write_parts(
        config.input_root,
        "lineitem",
        [pa.table({"l_orderkey": [10, 10, 20]}), pa.table({"l_orderkey": [30, 99]})],
    )

    output = build_index_npz(config, "tpch", 1, "lineitem", "orders")

    with np.load(output, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(
            artifact["__gather_idx_to_orders__"], [0, 0, 2, 1, -1]
        )


def test_each_builder_emits_an_independent_metadata_sidecar(tmp_path):
    config = _config(tmp_path)
    _write_parts(
        config.input_root,
        "region",
        [pa.table({"r_regionkey": [0, 1], "r_name": ["AFRICA", "AMERICA"]})],
    )

    outputs = [
        build_column_npz(config, "tpch", 1, "region", "r_regionkey"),
        build_string_npz(config, "tpch", 1, "region", "r_name"),
        build_system_npz(config, "tpch", 1, "region", "__valid_mask__"),
    ]

    for output in outputs:
        assert output.with_suffix(".meta.json").is_file()


def test_decimal_statistics_are_json_serializable(tmp_path):
    config = _config(tmp_path)
    _write_parts(
        config.input_root,
        "lineitem",
        [
            pa.table(
                {
                    "l_extendedprice": pa.array(
                        [Decimal("1.25"), Decimal("9.75")],
                        type=pa.decimal128(15, 2),
                    )
                }
            )
        ],
    )

    output = build_column_npz(config, "tpch", 1, "lineitem", "l_extendedprice")

    metadata = json.loads(output.with_suffix(".meta.json").read_text())
    assert metadata["minimum"] == "1.25"
    assert metadata["maximum"] == "9.75"
