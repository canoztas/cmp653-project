"""Tests for budget ledger."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.budget import AllocationStrategy, BudgetExhausted, BudgetLedger
from dpdb.parser import parse_query


class TestNaiveBudget:
    def test_basic_allocation(self):
        ledger = BudgetLedger(10.0, AllocationStrategy.NAIVE)
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        eps = ledger.allocate(q, 2.0)
        assert eps == 2.0
        assert ledger.remaining == 8.0

    def test_exhaustion(self):
        ledger = BudgetLedger(5.0, AllocationStrategy.NAIVE)
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        ledger.allocate(q, 3.0)
        ledger.allocate(q, 2.0)
        with pytest.raises(BudgetExhausted):
            ledger.allocate(q, 0.1)

    def test_no_caching(self):
        ledger = BudgetLedger(10.0, AllocationStrategy.NAIVE)
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        assert ledger.try_cache(q) is None


class TestWorkloadAwareBudget:
    def test_cache_hit(self):
        ledger = BudgetLedger(10.0, AllocationStrategy.WORKLOAD_AWARE)
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        ledger.allocate(q, 1.0)
        ledger.store_result(q, ["count"], [(42,)], 1.0)

        # Same query again should hit cache
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        cached = ledger.try_cache(q2)
        assert cached is not None
        assert cached.rows == [(42,)]
        assert ledger.consumed_epsilon == 1.0  # no additional cost

    def test_different_params_no_cache(self):
        ledger = BudgetLedger(10.0, AllocationStrategy.WORKLOAD_AWARE)
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        ledger.allocate(q1, 1.0)
        ledger.store_result(q1, ["count"], [(42,)], 1.0)

        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        cached = ledger.try_cache(q2)
        assert cached is None

    def test_summary(self):
        ledger = BudgetLedger(10.0, AllocationStrategy.WORKLOAD_AWARE)
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        ledger.allocate(q, 1.0)
        ledger.store_result(q, ["count"], [(100,)], 1.0)
        ledger.try_cache(parse_query("SELECT COUNT(*) FROM lineitem"))

        summary = ledger.summary()
        assert summary["cache_hits"] == 1
        assert summary["total_queries"] == 2
        assert summary["consumed_epsilon"] == 1.0
