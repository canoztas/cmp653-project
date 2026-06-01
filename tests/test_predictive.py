"""Tests for the predictive budget allocator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.parser import parse_query
from dpdb.predictive import PredictiveAllocator, PredictiveConfig


def _q(flag):
    return parse_query(f"SELECT COUNT(*) FROM lineitem WHERE l_returnflag = '{flag}'")


class TestPredictiveAllocator:
    def test_warmup_uses_average(self):
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=10.0, k_total=100, warmup_fraction=0.1, min_warmup=5,
        ))
        eps = alloc.next_epsilon(_q("R"))
        # During warmup: eps = total / k_total = 0.1
        assert abs(eps - 0.1) < 1e-6

    def test_adapts_after_warmup(self):
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=10.0, k_total=20, warmup_fraction=0.25,
        ))
        # Run through 5-query warmup with the same template
        for _ in range(5):
            eps = alloc.next_epsilon(_q("R"))
            alloc.note_release(_q("R"), eps)
        # After warmup, predictor sees 1 unique template; predicted remaining
        # unique ~ 1, so eps should be large (remaining budget / 1)
        eps_after = alloc.next_epsilon(_q("R"))
        assert eps_after > 0.5  # got bigger than warmup eps

    def test_budget_never_exceeded(self):
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=5.0, k_total=50,
        ))
        consumed = 0.0
        for i in range(50):
            eps = alloc.next_epsilon(_q(["R", "A", "N"][i % 3]))
            consumed += eps
            alloc.note_release(_q(["R", "A", "N"][i % 3]), eps)
        assert consumed <= 5.0 + 1e-6

    def test_summary_reports_state(self):
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=10.0, k_total=30,
        ))
        for flag in ["R", "A", "N", "R", "R"]:
            eps = alloc.next_epsilon(_q(flag))
            alloc.note_release(_q(flag), eps)
        s = alloc.summary()
        assert s["total_queries"] == 5
        assert s["unique_templates"] == 1  # all same template, different params
        assert s["consumed_epsilon"] > 0
        assert s["remaining_budget"] < 10.0

    def test_zero_when_exhausted(self):
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=0.5, k_total=1000, min_warmup=1, warmup_fraction=0.001,
        ))
        # Spend it all in 1 query
        eps = alloc.next_epsilon(_q("R"))
        alloc.note_release(_q("R"), eps + 0.6)  # force overspend
        # Now allocator should return 0
        eps2 = alloc.next_epsilon(_q("A"))
        assert eps2 == 0.0


# structurally-distinct templates (different aggregate/column), so the allocator
# actually sees multiple unique templates -- needed to exercise the estimators
_DISTINCT = [
    parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'"),
    parse_query("SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'R'"),
    parse_query("SELECT AVG(l_extendedprice) FROM lineitem"),
    parse_query("SELECT COUNT(*) FROM lineitem WHERE l_linestatus = 'O'"),
]


class TestEstimatorOption:
    def test_estimator_default_is_plugin(self):
        assert PredictiveConfig(total_budget=10.0, k_total=100).estimator == "plugin"

    def test_smoothed_gt_runs_and_respects_budget(self):
        # the smoothed-GT estimator is now selectable in the live allocator
        alloc = PredictiveAllocator(PredictiveConfig(
            total_budget=10.0, k_total=40, warmup_fraction=0.1, min_warmup=4,
            estimator="smoothed_gt",
        ))
        consumed = 0.0
        for i in range(40):
            q = _DISTINCT[i % len(_DISTINCT)]
            eps = alloc.next_epsilon(q)
            assert eps == eps and eps >= 0.0  # finite, non-negative (no NaN)
            consumed += eps
            alloc.note_release(q, eps)
        assert consumed <= 10.0 + 1e-6              # budget never exceeded
        assert alloc.summary()["unique_templates"] >= 2  # really saw distinct templates

    def test_smoothed_gt_uhat_is_finite_and_bounded(self):
        # the SGT U_hat must be a finite, positive value within the horizon;
        # at extreme extrapolation SGT diverges, so the allocator falls back to
        # the plug-in -- either way the result stays usable.
        a = PredictiveAllocator(PredictiveConfig(
            total_budget=10.0, k_total=100, estimator="smoothed_gt"))
        for i in range(8):
            a.next_epsilon(_DISTINCT[i % len(_DISTINCT)])
        u = a._predicted_total_unique()
        assert u == u                  # not NaN
        assert 0 < u <= 100 + 1e-6     # positive, within k_total
