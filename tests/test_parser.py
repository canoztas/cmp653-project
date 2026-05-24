"""Tests for SQL parser and validation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.parser import ParseError, parse_query


class TestValidQueries:
    def test_simple_count(self):
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        assert q.table == "lineitem"
        assert len(q.aggregates) == 1
        assert q.aggregates[0].func == "COUNT"
        assert q.aggregates[0].column is None

    def test_count_with_where(self):
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        assert q.table == "lineitem"
        assert q.where_clause is not None
        assert len(q.where_predicates) == 1

    def test_sum(self):
        q = parse_query("SELECT SUM(l_quantity) FROM lineitem")
        assert q.aggregates[0].func == "SUM"
        assert q.aggregates[0].column == "l_quantity"

    def test_avg(self):
        q = parse_query("SELECT AVG(l_extendedprice) FROM lineitem")
        assert q.aggregates[0].func == "AVG"
        assert q.aggregates[0].column == "l_extendedprice"

    def test_group_by(self):
        q = parse_query(
            "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"
        )
        assert q.group_by == ["l_returnflag"]
        assert len(q.aggregates) == 1

    def test_multiple_aggregates(self):
        q = parse_query("SELECT COUNT(*), SUM(l_quantity), AVG(l_extendedprice) FROM lineitem")
        assert len(q.aggregates) == 3

    def test_where_and(self):
        q = parse_query(
            "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R' AND l_quantity > 10"
        )
        assert len(q.where_predicates) == 2


class TestInvalidQueries:
    def test_update_rejected(self):
        with pytest.raises(ParseError, match="SELECT"):
            parse_query("UPDATE lineitem SET l_quantity = 0")

    def test_join_rejected(self):
        with pytest.raises(ParseError, match="JOIN"):
            parse_query(
                "SELECT COUNT(*) FROM lineitem JOIN orders ON l_orderkey = o_orderkey"
            )

    def test_subquery_rejected(self):
        with pytest.raises(ParseError, match="Subquer"):
            parse_query(
                "SELECT COUNT(*) FROM lineitem WHERE l_orderkey IN (SELECT o_orderkey FROM orders)"
            )

    def test_no_aggregate_rejected(self):
        with pytest.raises(ParseError, match="aggregate"):
            parse_query("SELECT l_quantity FROM lineitem")

    def test_non_aggregate_select_without_group_by_rejected(self):
        # Bug reported by Codex review: SELECT col, COUNT(*) without GROUP BY
        # used to be silently accepted. Should now raise.
        with pytest.raises(ParseError, match="GROUP BY"):
            parse_query("SELECT l_returnflag, COUNT(*) FROM lineitem")

    def test_non_aggregate_select_in_group_by_allowed(self):
        # The same query with proper GROUP BY must still parse.
        q = parse_query(
            "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"
        )
        assert q.group_by == ["l_returnflag"]
        assert len(q.aggregates) == 1

    def test_non_aggregate_column_not_in_group_by_rejected(self):
        # Selecting one column but grouping by a different one is invalid.
        with pytest.raises(ParseError, match="not in GROUP BY"):
            parse_query(
                "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_shipmode"
            )
