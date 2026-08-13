"""Declarative workload metadata; conversion logic contains no table branches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StringSpec:
    mode: str = "dictionary"
    tokenize: bool = False
    legacy_pool: bool = False


@dataclass(frozen=True)
class RelationshipSpec:
    name: str
    child_table: str
    child_columns: tuple[str, ...]
    parent_table: str
    parent_columns: tuple[str, ...]
    output_name: str | None = None

    def __post_init__(self) -> None:
        if not self.child_columns or len(self.child_columns) != len(
            self.parent_columns
        ):
            raise ValueError("PK-FK relationships require equally sized key tuples")

    @property
    def array_name(self) -> str:
        role = self.output_name or self.name
        return f"__gather_idx_to_{role}__"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    strings: Mapping[tuple[str, str], StringSpec] = field(default_factory=dict)
    relationships: Mapping[tuple[str, str], RelationshipSpec] = field(
        default_factory=dict
    )

    def string_spec(self, table: str, column: str) -> StringSpec:
        return self.strings.get((table, column), StringSpec())

    def relationship(self, child_table: str, name: str) -> RelationshipSpec:
        try:
            return self.relationships[(child_table, name)]
        except KeyError as error:
            choices = sorted(
                relation_name
                for table, relation_name in self.relationships
                if table == child_table
            )
            raise ValueError(
                f"unknown {self.name} relationship {child_table}.{name}; "
                f"expected one of {choices}"
            ) from error


def _relationship(
    child: str,
    name: str,
    child_columns: str | tuple[str, ...],
    parent: str,
    parent_columns: str | tuple[str, ...],
) -> RelationshipSpec:
    if isinstance(child_columns, str):
        child_columns = (child_columns,)
    if isinstance(parent_columns, str):
        parent_columns = (parent_columns,)
    return RelationshipSpec(name, child, child_columns, parent, parent_columns)


_TPCH_RELATIONS = [
    _relationship("nation", "region", "n_regionkey", "region", "r_regionkey"),
    _relationship("supplier", "nation", "s_nationkey", "nation", "n_nationkey"),
    _relationship("customer", "nation", "c_nationkey", "nation", "n_nationkey"),
    _relationship("orders", "customer", "o_custkey", "customer", "c_custkey"),
    _relationship("partsupp", "part", "ps_partkey", "part", "p_partkey"),
    _relationship("partsupp", "supplier", "ps_suppkey", "supplier", "s_suppkey"),
    _relationship("lineitem", "orders", "l_orderkey", "orders", "o_orderkey"),
    _relationship("lineitem", "part", "l_partkey", "part", "p_partkey"),
    _relationship("lineitem", "supplier", "l_suppkey", "supplier", "s_suppkey"),
    _relationship(
        "lineitem",
        "partsupp",
        ("l_partkey", "l_suppkey"),
        "partsupp",
        ("ps_partkey", "ps_suppkey"),
    ),
]

_ROW_DICTIONARY_COLUMNS = {
    ("supplier", "s_name"),
    ("supplier", "s_address"),
    ("supplier", "s_phone"),
    ("customer", "c_name"),
    ("customer", "c_address"),
    ("customer", "c_phone"),
    ("customer", "c_comment"),
}
_TOKEN_COLUMNS = {
    ("orders", "o_comment"),
    ("part", "p_name"),
    ("supplier", "s_comment"),
}
_LEGACY_POOL_COLUMNS = {
    ("customer", "c_phone"),
    ("orders", "o_comment"),
    ("part", "p_name"),
    ("part", "p_type"),
    ("supplier", "s_comment"),
}


def _tpcds_relations() -> list[RelationshipSpec]:
    """Standard TPC-DS dimension links, named by semantic FK role."""

    definitions = {
        "store_sales": {
            "sold_date": ("ss_sold_date_sk", "date_dim", "d_date_sk"),
            "sold_time": ("ss_sold_time_sk", "time_dim", "t_time_sk"),
            "item": ("ss_item_sk", "item", "i_item_sk"),
            "customer": ("ss_customer_sk", "customer", "c_customer_sk"),
            "customer_demographics": (
                "ss_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "household_demographics": (
                "ss_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "customer_address": (
                "ss_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "store": ("ss_store_sk", "store", "s_store_sk"),
            "promotion": ("ss_promo_sk", "promotion", "p_promo_sk"),
        },
        "store_returns": {
            "returned_date": ("sr_returned_date_sk", "date_dim", "d_date_sk"),
            "return_time": ("sr_return_time_sk", "time_dim", "t_time_sk"),
            "item": ("sr_item_sk", "item", "i_item_sk"),
            "customer": ("sr_customer_sk", "customer", "c_customer_sk"),
            "customer_demographics": (
                "sr_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "household_demographics": (
                "sr_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "customer_address": (
                "sr_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "store": ("sr_store_sk", "store", "s_store_sk"),
            "reason": ("sr_reason_sk", "reason", "r_reason_sk"),
        },
        "inventory": {
            "date": ("inv_date_sk", "date_dim", "d_date_sk"),
            "item": ("inv_item_sk", "item", "i_item_sk"),
            "warehouse": ("inv_warehouse_sk", "warehouse", "w_warehouse_sk"),
        },
    }
    shared_sales = {
        "catalog_sales": "cs",
        "web_sales": "ws",
    }
    for table, prefix in shared_sales.items():
        definitions[table] = {
            "sold_date": (f"{prefix}_sold_date_sk", "date_dim", "d_date_sk"),
            "sold_time": (f"{prefix}_sold_time_sk", "time_dim", "t_time_sk"),
            "ship_date": (f"{prefix}_ship_date_sk", "date_dim", "d_date_sk"),
            "bill_customer": (
                f"{prefix}_bill_customer_sk",
                "customer",
                "c_customer_sk",
            ),
            "bill_customer_demographics": (
                f"{prefix}_bill_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "bill_household_demographics": (
                f"{prefix}_bill_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "bill_address": (
                f"{prefix}_bill_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "ship_customer": (
                f"{prefix}_ship_customer_sk",
                "customer",
                "c_customer_sk",
            ),
            "ship_customer_demographics": (
                f"{prefix}_ship_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "ship_household_demographics": (
                f"{prefix}_ship_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "ship_address": (
                f"{prefix}_ship_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "ship_mode": (f"{prefix}_ship_mode_sk", "ship_mode", "sm_ship_mode_sk"),
            "warehouse": (f"{prefix}_warehouse_sk", "warehouse", "w_warehouse_sk"),
            "item": (f"{prefix}_item_sk", "item", "i_item_sk"),
            "promotion": (f"{prefix}_promo_sk", "promotion", "p_promo_sk"),
        }
    definitions["catalog_sales"].update(
        {
            "call_center": ("cs_call_center_sk", "call_center", "cc_call_center_sk"),
            "catalog_page": (
                "cs_catalog_page_sk",
                "catalog_page",
                "cp_catalog_page_sk",
            ),
        }
    )
    definitions["web_sales"].update(
        {
            "web_page": ("ws_web_page_sk", "web_page", "wp_web_page_sk"),
            "web_site": ("ws_web_site_sk", "web_site", "web_site_sk"),
        }
    )
    for table, prefix in {"catalog_returns": "cr", "web_returns": "wr"}.items():
        definitions[table] = {
            "returned_date": (f"{prefix}_returned_date_sk", "date_dim", "d_date_sk"),
            "returned_time": (f"{prefix}_returned_time_sk", "time_dim", "t_time_sk"),
            "item": (f"{prefix}_item_sk", "item", "i_item_sk"),
            "refunded_customer": (
                f"{prefix}_refunded_customer_sk",
                "customer",
                "c_customer_sk",
            ),
            "refunded_customer_demographics": (
                f"{prefix}_refunded_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "refunded_household_demographics": (
                f"{prefix}_refunded_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "refunded_address": (
                f"{prefix}_refunded_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "returning_customer": (
                f"{prefix}_returning_customer_sk",
                "customer",
                "c_customer_sk",
            ),
            "returning_customer_demographics": (
                f"{prefix}_returning_cdemo_sk",
                "customer_demographics",
                "cd_demo_sk",
            ),
            "returning_household_demographics": (
                f"{prefix}_returning_hdemo_sk",
                "household_demographics",
                "hd_demo_sk",
            ),
            "returning_address": (
                f"{prefix}_returning_addr_sk",
                "customer_address",
                "ca_address_sk",
            ),
            "reason": (f"{prefix}_reason_sk", "reason", "r_reason_sk"),
        }
    definitions["catalog_returns"].update(
        {
            "catalog_sales": (
                ("cr_order_number", "cr_item_sk"),
                "catalog_sales",
                ("cs_order_number", "cs_item_sk"),
            ),
            "call_center": ("cr_call_center_sk", "call_center", "cc_call_center_sk"),
            "catalog_page": (
                "cr_catalog_page_sk",
                "catalog_page",
                "cp_catalog_page_sk",
            ),
            "ship_mode": ("cr_ship_mode_sk", "ship_mode", "sm_ship_mode_sk"),
            "warehouse": ("cr_warehouse_sk", "warehouse", "w_warehouse_sk"),
        }
    )
    definitions["web_returns"]["web_page"] = (
        "wr_web_page_sk",
        "web_page",
        "wp_web_page_sk",
    )
    definitions["web_returns"]["web_sales"] = (
        ("wr_order_number", "wr_item_sk"),
        "web_sales",
        ("ws_order_number", "ws_item_sk"),
    )
    definitions["store_returns"]["store_sales"] = (
        ("sr_ticket_number", "sr_item_sk"),
        "store_sales",
        ("ss_ticket_number", "ss_item_sk"),
    )
    definitions.update(
        {
            "customer": {
                "current_customer_demographics": (
                    "c_current_cdemo_sk",
                    "customer_demographics",
                    "cd_demo_sk",
                ),
                "current_household_demographics": (
                    "c_current_hdemo_sk",
                    "household_demographics",
                    "hd_demo_sk",
                ),
                "current_address": (
                    "c_current_addr_sk",
                    "customer_address",
                    "ca_address_sk",
                ),
            },
            "household_demographics": {
                "income_band": ("hd_income_band_sk", "income_band", "ib_income_band_sk")
            },
            "promotion": {
                "item": ("p_item_sk", "item", "i_item_sk"),
            },
            "store": {
                "closed_date": ("s_closed_date_sk", "date_dim", "d_date_sk"),
            },
            "call_center": {
                "open_date": ("cc_open_date_sk", "date_dim", "d_date_sk"),
                "closed_date": ("cc_closed_date_sk", "date_dim", "d_date_sk"),
            },
            "web_site": {
                "open_date": ("web_open_date_sk", "date_dim", "d_date_sk"),
                "close_date": ("web_close_date_sk", "date_dim", "d_date_sk"),
            },
            "web_page": {
                "creation_date": (
                    "wp_creation_date_sk",
                    "date_dim",
                    "d_date_sk",
                ),
                "access_date": ("wp_access_date_sk", "date_dim", "d_date_sk"),
            },
        }
    )
    return [
        _relationship(table, role, child_col, parent, parent_col)
        for table, roles in definitions.items()
        for role, (child_col, parent, parent_col) in roles.items()
    ]


TPCH = WorkloadSpec(
    "tpch",
    strings={
        key: StringSpec(
            mode="row_dictionary" if key in _ROW_DICTIONARY_COLUMNS else "dictionary",
            tokenize=key in _TOKEN_COLUMNS,
            legacy_pool=key in _LEGACY_POOL_COLUMNS,
        )
        for key in _ROW_DICTIONARY_COLUMNS | _TOKEN_COLUMNS | _LEGACY_POOL_COLUMNS
    },
    relationships={(item.child_table, item.name): item for item in _TPCH_RELATIONS},
)

_TPCDS_RELATIONS = _tpcds_relations()
TPCDS = WorkloadSpec(
    "tpcds",
    relationships={(item.child_table, item.name): item for item in _TPCDS_RELATIONS},
)

WORKLOADS: dict[str, WorkloadSpec] = {"tpch": TPCH, "tpcds": TPCDS}


def get_workload(
    name: str, overrides: Mapping[str, WorkloadSpec] | None = None
) -> WorkloadSpec:
    normalized = name.lower().replace("-", "")
    registry = dict(WORKLOADS)
    if overrides:
        registry.update(overrides)
    try:
        return registry[normalized]
    except KeyError as error:
        raise ValueError(
            f"unknown workload {name!r}; expected one of {sorted(registry)}"
        ) from error
