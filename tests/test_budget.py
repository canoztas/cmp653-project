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

    def test_unique_templates_counts_exact_species(self):
        # Three queries that share a structural template but differ in the WHERE
        # literal are THREE distinct exact species, each charged epsilon. The
        # summary must report 3, not 1 (structural-template under-count).
        ledger = BudgetLedger(100.0, AllocationStrategy.WORKLOAD_AWARE)
        for flag in ["R", "A", "N"]:
            q = parse_query(f"SELECT COUNT(*) FROM lineitem WHERE l_returnflag = '{flag}'")
            ledger.allocate(q, 1.0)
            ledger.store_result(q, ["c"], [(1,)], 1.0)
        assert ledger.summary()["unique_templates"] == 3
        assert ledger.consumed_epsilon == 3.0


class TestTemporalUpdateSeed:
    """Regression: the update-invalidation RNG must be seeded per ledger so trials
    are (a) reproducible for a fixed seed and (b) independent across seeds. A fixed
    Random(42) made every trial share one update realization."""

    def _evictions(self, seed: int) -> int:
        ledger = BudgetLedger(
            1e9, AllocationStrategy.WORKLOAD_AWARE,
            staleness_tolerance=1e9,        # no age-based expiry; updates only
            update_rate=0.5, update_invalidation_prob=0.5, update_seed=seed,
        )
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        for _ in range(200):
            if ledger.try_cache(q) is None:        # miss (or just-invalidated): re-noise
                ledger.allocate(q, 1.0)
                ledger.store_result(q, ["c"], [(1,)], 1.0)
        return ledger.summary()["update_evictions"]

    def test_same_seed_is_deterministic(self):
        assert self._evictions(7) == self._evictions(7)

    def test_independent_seeds_vary(self):
        # independent update streams -> not all seeds collapse to one value
        assert len({self._evictions(s) for s in range(6)}) > 1


class TestBudgetAwareReReleaseCeiling:
    """Proposition (budget-aware re-release, prop:barr in the paper): a budget-gated
    sequence of eps_q releases spends at most B and admits exactly
    min(R, floor(B/eps_q)) paid releases for any number R of requested releases.
    This pins the proven temporal ceiling empirically (the paper's provable
    upper bound, as opposed to the magnitude planning estimate)."""

    def _paid_releases(self, total_budget, eps_q, requested):
        ledger = BudgetLedger(total_budget, AllocationStrategy.NAIVE)
        paid = 0
        for i in range(requested):
            q = parse_query(f"SELECT COUNT(*) FROM lineitem WHERE l_linenumber = {i}")
            try:
                ledger.allocate(q, eps_q)
                paid += 1
            except BudgetExhausted:
                break
        return paid, ledger

    def test_ceiling_when_requests_exceed_capacity(self):
        # B=10, eps_q=1, R=25 requested -> exactly floor(B/eps_q)=10 paid, spend<=B
        paid, ledger = self._paid_releases(10.0, 1.0, 25)
        assert paid == 10                       # = floor(B/eps_q)
        assert ledger.remaining >= -1e-9
        assert (10.0 - ledger.remaining) <= 10.0 + 1e-9   # cumulative spend never exceeds B

    def test_all_succeed_when_requests_below_capacity(self):
        # R=6 < floor(B/eps_q)=10 -> all 6 succeed, count = min(R, floor(B/eps_q)) = R
        paid, ledger = self._paid_releases(10.0, 1.0, 6)
        assert paid == 6
        assert ledger.remaining == pytest.approx(4.0)
