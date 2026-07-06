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

    def test_inner_fk_join_count_accepted(self):
        # A single 2-table inner equi-join COUNT now parses; the privacy-relevant
        # d_max sensitivity is resolved against config in the analyzer, not here.
        q = parse_query(
            "SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey"
        )
        assert q.is_join
        assert q.join_table == "lineitem"
        assert q.tables == ["orders", "lineitem"]

    def test_sum_over_join_rejected(self):
        with pytest.raises(ParseError, match="COUNT is supported over a JOIN"):
            parse_query(
                "SELECT SUM(l_quantity) FROM orders JOIN lineitem ON l_orderkey = o_orderkey"
            )

    def test_outer_join_rejected(self):
        with pytest.raises(ParseError, match="INNER FK joins"):
            parse_query(
                "SELECT COUNT(*) FROM orders LEFT JOIN lineitem ON l_orderkey = o_orderkey"
            )

    def test_multiple_joins_rejected(self):
        with pytest.raises(ParseError, match="one JOIN"):
            parse_query(
                "SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey "
                "JOIN customer ON o_custkey = c_custkey"
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


class TestSensitivityBypassRejected:
    """Adversarial regression: crafted SQL that would slip past sensitivity
    analysis (wrapping/windowing an aggregate, or amplifying contribution) must
    be rejected at parse time, BEFORE any budget is spent."""

    def test_arithmetic_wrapped_aggregate_rejected(self):
        # 1000*SUM(age): true sensitivity is 1000*B_c, not B_c.
        with pytest.raises(ParseError, match="direct SELECT item"):
            parse_query("SELECT 1000 * SUM(age) FROM adult")

    def test_window_count_rejected(self):
        # COUNT(*) OVER () returns the true count in every one of N rows.
        with pytest.raises(ParseError):
            parse_query("SELECT COUNT(*) OVER () FROM adult")

    def test_window_sum_rejected(self):
        with pytest.raises(ParseError):
            parse_query("SELECT SUM(age) OVER (PARTITION BY sex) FROM adult")

    def test_coalesce_wrapped_aggregate_rejected(self):
        with pytest.raises(ParseError, match="direct SELECT item"):
            parse_query("SELECT COALESCE(SUM(age), 0) FROM adult")

    def test_filter_aggregate_rejected(self):
        with pytest.raises(ParseError):
            parse_query("SELECT SUM(age) FILTER (WHERE sex = 'Male') FROM adult")

    def test_cte_self_union_rejected(self):
        # Each row contributes twice; sensitivity-1 accounting would be wrong.
        with pytest.raises(ParseError, match="WITH|CTE"):
            parse_query(
                "WITH x AS (SELECT * FROM adult UNION ALL SELECT * FROM adult) "
                "SELECT COUNT(*) FROM x")

    def test_top_level_union_rejected(self):
        with pytest.raises(ParseError):
            parse_query("SELECT COUNT(*) FROM adult UNION ALL SELECT COUNT(*) FROM adult")

    def test_select_distinct_rejected(self):
        with pytest.raises(ParseError, match="DISTINCT"):
            parse_query("SELECT DISTINCT sex FROM adult")


class TestTopKBoundsRejected:
    """Adversarial regression for the top-k parser surface."""

    def test_order_by_ordinal_out_of_range_rejected(self):
        # 2-column result, ORDER BY 3 -> would index past the row -> IndexError
        # AFTER budget spend. Must reject at parse.
        with pytest.raises(ParseError):
            parse_query("SELECT education, COUNT(*) FROM adult GROUP BY education ORDER BY 3")

    def test_order_by_zero_rejected(self):
        # ORDER BY 0 would map to Python index -1 (last column).
        with pytest.raises(ParseError):
            parse_query("SELECT education, COUNT(*) FROM adult GROUP BY education ORDER BY 0")

    def test_limit_on_group_key_rejected(self):
        # top-k must rank on an aggregate, not a public group key.
        with pytest.raises(ParseError, match="aggregate"):
            parse_query("SELECT sex, COUNT(*) FROM adult GROUP BY sex ORDER BY sex LIMIT 1")

    def test_offset_rejected(self):
        with pytest.raises(ParseError, match="OFFSET"):
            parse_query(
                "SELECT education, COUNT(*) AS c FROM adult GROUP BY education "
                "ORDER BY c DESC LIMIT 5 OFFSET 3")

    def test_valid_topk_still_parses(self):
        q = parse_query(
            "SELECT education, COUNT(*) AS c FROM adult GROUP BY education "
            "ORDER BY c DESC LIMIT 5")
        assert q.limit == 5 and q.order_by_position == 1


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
