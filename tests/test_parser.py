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

    def test_having_rejected(self):
        # HAVING filters groups on a data-dependent aggregate condition; it is an
        # unaccounted private selection and must be rejected like JOIN/subquery.
        with pytest.raises(ParseError, match="HAVING"):
            parse_query(
                "SELECT l_returnflag, COUNT(*) FROM lineitem "
                "GROUP BY l_returnflag HAVING COUNT(*) > 5"
            )

    def test_count_distinct_rejected(self):
        # COUNT(DISTINCT col) is outside the supported subset; reject rather than
        # stringify the DISTINCT node into a bogus column name.
        with pytest.raises(ParseError, match="DISTINCT"):
            parse_query("SELECT COUNT(DISTINCT l_returnflag) FROM lineitem")


class TestAggregatePosition:
    """Regression tests for the column-ordering privacy bug: each aggregate's
    SELECT-list position must be recorded so the noise mechanism targets the
    correct result column regardless of ordering."""

    def test_group_key_first(self):
        q = parse_query(
            "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"
        )
        assert len(q.aggregates) == 1
        assert q.aggregates[0].position == 1  # COUNT is the 2nd SELECT item

    def test_aggregate_first(self):
        # The dangerous ordering: aggregate before the group key.
        q = parse_query(
            "SELECT COUNT(*), l_returnflag FROM lineitem GROUP BY l_returnflag"
        )
        assert len(q.aggregates) == 1
        assert q.aggregates[0].position == 0  # COUNT is the 1st SELECT item

    def test_multiple_aggregate_positions(self):
        q = parse_query(
            "SELECT SUM(l_quantity), l_returnflag, COUNT(*) "
            "FROM lineitem GROUP BY l_returnflag"
        )
        positions = [a.position for a in q.aggregates]
        assert positions == [0, 2]  # SUM at 0, COUNT at 2 (group key at 1)
