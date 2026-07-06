"""Configuration loading and dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "tpch"
    user: str = "postgres"
    password: str = "postgres"


@dataclass
class PrivacyConfig:
    total_epsilon: float = 10.0
    default_query_epsilon: float = 1.0
    mechanism: str = "laplace"


@dataclass
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    column_bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    # Public foreign-key multiplicity bounds for supported FK joins:
    # fk_multiplicity[parent][child] = d_max = the maximum number of child rows a
    # single parent entity can match. It is the conservative global-sensitivity
    # constant for a COUNT over the parent-child join when the privacy unit is the
    # parent entity (removing one parent removes at most d_max joined rows). It is
    # a PUBLIC schema property (e.g. the TPC-H spec caps line-items/order at 7),
    # so naming it leaks nothing about the protected rows.
    fk_multiplicity: dict[str, dict[str, float]] = field(default_factory=dict)
    backend: str = "duckdb"  # "duckdb" or "postgres"
    duckdb_path: str = "data/dpdb.duckdb"

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "Config":
        if path is None:
            path = Path(__file__).parent.parent.parent / "config.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        db_raw = raw.get("database", {})
        priv_raw = raw.get("privacy", {})
        return cls(
            db=DBConfig(
                host=db_raw.get("host", "localhost"),
                port=db_raw.get("port", 5432),
                name=db_raw.get("name", "tpch"),
                user=db_raw.get("user", "postgres"),
                password=db_raw.get("password", "postgres"),
            ),
            privacy=PrivacyConfig(
                total_epsilon=priv_raw.get("total_epsilon", 10.0),
                default_query_epsilon=priv_raw.get("default_query_epsilon", 1.0),
                mechanism=priv_raw.get("mechanism", "laplace"),
            ),
            column_bounds=raw.get("column_bounds", {}),
            fk_multiplicity=raw.get("fk_multiplicity", {}),
            backend=raw.get("backend", "duckdb"),
            duckdb_path=raw.get("duckdb_path", "data/dpdb.duckdb"),
        )

    def get_bound(self, table: str, column: str) -> Optional[float]:
        return self.column_bounds.get(table, {}).get(column)

    def get_join_bound(self, table_a: str, table_b: str) -> Optional[tuple[str, str, float]]:
        """Return (parent, child, d_max) for a declared FK join between the two
        tables in EITHER order, or None if the pair is not a supported FK join.
        The privacy unit is the returned parent entity."""
        if table_b in self.fk_multiplicity.get(table_a, {}):
            return (table_a, table_b, float(self.fk_multiplicity[table_a][table_b]))
        if table_a in self.fk_multiplicity.get(table_b, {}):
            return (table_b, table_a, float(self.fk_multiplicity[table_b][table_a]))
        return None
