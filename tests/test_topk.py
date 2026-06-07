"""Tests for top-k support (ORDER BY <agg> [DESC|ASC] LIMIT k).

Top-k is a post-processing of the noised GROUP BY histogram: the middleware fetches
ALL groups, noises every count under parallel composition (cost eps_q, NOT k*eps_q),
then sorts by the NOISY value and truncates. Selecting on the true counts would leak
which groups crossed the cutoff, so these tests pin both the budget and the
selection-on-noise behaviour.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.parser import ParseError, parse_query

TOPK = "SELECT education, COUNT(*) AS c FROM adult GROUP BY education ORDER BY c DESC LIMIT 5"


class TestTopKParser:
    def test_parses_order_and_limit(self):
        p = parse_query(TOPK)
        assert p.limit == 5
        assert p.order_by_position == 1     # the COUNT column
        assert p.order_desc is True

    def test_order_by_count_star(self):
        p = parse_query("SELECT sex, COUNT(*) FROM adult GROUP BY sex ORDER BY COUNT(*) DESC LIMIT 2")
        assert p.limit == 2 and p.order_by_position == 1

    def test_asc(self):
        p = parse_query("SELECT sex, AVG(age) AS a FROM adult GROUP BY sex ORDER BY a ASC LIMIT 1")
        assert p.order_desc is False and p.limit == 1

    def test_no_limit_is_none(self):
        assert parse_query("SELECT COUNT(*) FROM adult").limit is None


class TestTopKMechanism:
    def _mw(self, mode, B=100.0):
        cfg = Config.from_yaml()
        cfg.privacy.total_epsilon = B
        return DPMiddleware(cfg, mode=mode)

    def test_returns_k_rows(self):
        r = self._mw(ExecutionMode.WORKLOAD_DP).execute(TOPK, epsilon=1.0)
        assert r.error is None
        assert len(r.rows) == 5

    def test_costs_same_as_group_by(self):
        # post-processing: top-k charges eps_q, NOT k*eps_q
        r = self._mw(ExecutionMode.WORKLOAD_DP).execute(TOPK, epsilon=1.0)
        assert abs(r.epsilon_used - 1.0) < 1e-9

    def test_rows_sorted_descending_by_noisy_value(self):
        r = self._mw(ExecutionMode.WORKLOAD_DP).execute(TOPK, epsilon=1.0)
        vals = [row[1] for row in r.rows]
        assert vals == sorted(vals, reverse=True)

    def test_selection_uses_noisy_not_true_counts(self):
        # with a tiny eps (huge noise) the released top-k SET deviates from the true
        # top-5 in at least some runs -> proves the selection is on the noised counts
        mw = self._mw(ExecutionMode.NAIVE_DP, B=10000.0)
        true_top5 = {row[0] for row in
                     DPMiddleware(Config.from_yaml(), mode=ExecutionMode.EXACT)
                     .execute(TOPK).rows}
        deviated = 0
        for _ in range(30):
            got = {row[0] for row in mw.execute(TOPK, epsilon=0.001).rows}
            if got != true_top5:
                deviated += 1
        assert deviated > 0   # the set is noise-dependent, not a true-count selection

    def test_exact_mode_is_true_topk(self):
        r = DPMiddleware(Config.from_yaml(), mode=ExecutionMode.EXACT).execute(TOPK)
        vals = [row[1] for row in r.rows]
        assert len(r.rows) == 5 and vals == sorted(vals, reverse=True)


class TestOrderByLimitLeak:
    """Regression: ORDER BY (with or without LIMIT) must rank on the NOISED value,
    and a LIMIT without an explicit ORDER BY is rejected (ill-defined top-k)."""

    def _mw(self, mode, B=10000.0):
        cfg = Config.from_yaml()
        cfg.privacy.total_epsilon = B
        return DPMiddleware(cfg, mode=mode)

    def test_limit_without_order_by_is_rejected(self):
        with pytest.raises(ParseError):
            parse_query("SELECT sex, COUNT(*) FROM adult GROUP BY sex LIMIT 2")

    def test_order_by_without_limit_returns_all_rows_sorted_on_noise(self):
        # No LIMIT: every group is returned, ordered by the NOISED count.
        q = "SELECT education, COUNT(*) AS c FROM adult GROUP BY education ORDER BY c DESC"
        r = self._mw(ExecutionMode.WORKLOAD_DP).execute(q, epsilon=1.0)
        assert r.error is None
        vals = [row[1] for row in r.rows]
        assert vals == sorted(vals, reverse=True)
        # all groups present (nothing truncated)
        n_groups = len(DPMiddleware(Config.from_yaml(), mode=ExecutionMode.EXACT)
                       .execute("SELECT education, COUNT(*) FROM adult GROUP BY education").rows)
        assert len(r.rows) == n_groups

    def test_order_by_without_limit_is_noise_dependent(self):
        # With huge noise the released ordering deviates from the true-count order
        # in at least some runs -> the sort is on the noised value, not the true one.
        q = "SELECT education, COUNT(*) AS c FROM adult GROUP BY education ORDER BY c ASC"
        mw = self._mw(ExecutionMode.NAIVE_DP)
        true_order = [row[0] for row in
                      DPMiddleware(Config.from_yaml(), mode=ExecutionMode.EXACT)
                      .execute(q).rows]
        deviated = 0
        for _ in range(40):
            got = [row[0] for row in mw.execute(q, epsilon=0.0005).rows]
            if got != true_order:
                deviated += 1
        assert deviated > 0
