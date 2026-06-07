"""Regression tests for SUM/AVG domain clipping in the executed SQL.

The middleware clamps each summed/averaged value to its public bound [0, B_c]
INSIDE the executed SQL, so the realized sensitivity can never exceed Delta f = B_c.
Two correctness requirements are pinned here:
  1. Out-of-bound values are clamped to B_c (the point of the mechanism).
  2. SQL NULL is preserved (NOT turned into B_c): GREATEST/LEAST treat NULL as a
     missing operand, so a naive GREATEST(LEAST(c,B),0) would corrupt SUM/AVG.
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.parser import parse_query


def _clamped_sql(select_sql: str, table: str, col: str, bound: float) -> str:
    """Render the clamped SQL the middleware would execute (no noise)."""
    cfg = Config.from_yaml()
    cfg.column_bounds = {table: {col: bound}}
    mw = DPMiddleware(cfg, mode=ExecutionMode.EXACT)
    base = parse_query(select_sql).ast.copy()
    assert mw._clamp_aggregate_columns(base, table) is True
    return base.sql(dialect="postgres")


def _scalar(agg: str) -> float:
    con = duckdb.connect()
    con.execute("CREATE TABLE t(x DOUBLE)")
    con.execute("INSERT INTO t VALUES (-5),(5),(150),(NULL)")
    sql = _clamped_sql(f"SELECT {agg} FROM t", "t", "x", 100.0)
    return con.execute(sql).fetchone()[0]


class TestClampNullSemantics:
    def test_sum_clamps_and_preserves_null(self):
        # -5->0, 5->5, 150->100, NULL stays NULL (excluded) => 105 (NOT 205)
        assert _scalar("SUM(x)") == 105

    def test_avg_clamps_and_preserves_null(self):
        # mean over the 3 non-null clamped values {0,5,100} = 35 (NOT 51.25)
        assert _scalar("AVG(x)") == 35


class TestClampOutOfBound:
    def test_value_above_bound_is_clamped(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE t(x DOUBLE)")
        con.execute("INSERT INTO t VALUES (150),(200)")
        sql = _clamped_sql("SELECT SUM(x) FROM t", "t", "x", 100.0)
        assert con.execute(sql).fetchone()[0] == 200  # both clamped to 100

    def test_in_bound_values_unchanged(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE t(x DOUBLE)")
        con.execute("INSERT INTO t VALUES (10),(20),(30)")
        sql = _clamped_sql("SELECT SUM(x) FROM t", "t", "x", 100.0)
        assert con.execute(sql).fetchone()[0] == 60  # unchanged

    def test_clamped_sql_is_null_guarded(self):
        # the generated SQL must guard NULL explicitly
        sql = _clamped_sql("SELECT SUM(x) FROM t", "t", "x", 100.0)
        assert "IS NULL" in sql.upper() and "GREATEST" in sql.upper()
