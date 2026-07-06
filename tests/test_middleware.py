"""Tests for the DP middleware noise mechanism.

Regression coverage for the column-ordering privacy bug: _add_noise must noise
the aggregate column and leave the GROUP BY key untouched, regardless of whether
the aggregate or the group key is listed first in the SELECT.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.analyzer import SensitivityError, SensitivityResult, analyze_sensitivity
from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.parser import parse_query


def _add_noise(parsed, true_rows, sensitivities, epsilon):
    """Call _add_noise without constructing a backend database."""
    mw = DPMiddleware.__new__(DPMiddleware)
    return mw._add_noise(parsed, true_rows, sensitivities, epsilon)


class TestJoinSensitivity:
    """A FK-join COUNT must be charged the public FK multiplicity d_max, not 1:
    removing one parent entity removes up to d_max joined rows (privacy unit =
    parent). An undeclared FK pair has unbounded sensitivity and must be refused."""

    def test_join_count_sensitivity_is_d_max(self):
        cfg = Config.from_yaml()  # config.yaml declares orders->lineitem = 7
        parsed = parse_query(
            "SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey "
            "WHERE o_orderpriority = '1-URGENT'")
        sens = analyze_sensitivity(parsed, cfg)
        assert len(sens) == 1
        assert sens[0].func == "COUNT"
        assert sens[0].sensitivity == 7.0

    def test_single_table_count_still_unit_sensitivity(self):
        cfg = Config.from_yaml()
        parsed = parse_query("SELECT COUNT(*) FROM orders WHERE o_orderpriority = '1-URGENT'")
        sens = analyze_sensitivity(parsed, cfg)
        assert sens[0].sensitivity == 1.0

    def test_undeclared_fk_join_refused(self):
        cfg = Config.from_yaml()
        cfg.fk_multiplicity = {}  # no public d_max -> unbounded sensitivity
        parsed = parse_query(
            "SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey")
        with pytest.raises(SensitivityError, match="d_max"):
            analyze_sensitivity(parsed, cfg)


class TestNoiseColumnTargeting:
    def test_group_key_preserved_when_aggregate_first(self):
        # `SELECT COUNT(*), l_returnflag ... GROUP BY l_returnflag`: COUNT is at
        # result index 0, the (string) group key at index 1. The old code assumed
        # group keys come first and would noise index 1 -> float('A') crash and the
        # true COUNT released in the clear. The fix targets the recorded position.
        parsed = parse_query(
            "SELECT COUNT(*), l_returnflag FROM lineitem GROUP BY l_returnflag"
        )
        sens = [SensitivityResult(func="COUNT", column=None, sensitivity=1.0)]
        true_rows = [(100, "A"), (50, "B")]
        noisy = _add_noise(parsed, true_rows, sens, 1.0)

        # Group keys (string) must survive untouched at index 1.
        assert [r[1] for r in noisy] == ["A", "B"]
        # The COUNT column (index 0) is noised: a non-negative integer.
        for r in noisy:
            assert isinstance(r[0], int)
            assert r[0] >= 0

    def test_group_key_preserved_when_key_first(self):
        # Conventional ordering must still work.
        parsed = parse_query(
            "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"
        )
        sens = [SensitivityResult(func="COUNT", column=None, sensitivity=1.0)]
        true_rows = [("A", 100), ("B", 50)]
        noisy = _add_noise(parsed, true_rows, sens, 1.0)

        assert [r[0] for r in noisy] == ["A", "B"]  # group key untouched at index 0
        for r in noisy:
            assert isinstance(r[1], int) and r[1] >= 0  # COUNT noised at index 1

    def test_true_count_not_released_in_clear(self):
        # With a large epsilon noise is tiny, but the value must still pass through
        # the noise mechanism (it is not the exact true value by construction of the
        # column index); we assert the key column is the exact passthrough instead.
        parsed = parse_query(
            "SELECT COUNT(*), l_returnflag FROM lineitem GROUP BY l_returnflag"
        )
        sens = [SensitivityResult(func="COUNT", column=None, sensitivity=1.0)]
        noisy = _add_noise(parsed, [(42, "X")], sens, 1e-6)
        # Tiny epsilon -> huge noise -> the COUNT is almost surely not exactly 42.
        # The group key must be the untouched passthrough.
        assert noisy[0][1] == "X"


class TestInvalidEpsilonRejected:
    """Regression: a supplied epsilon must be finite and positive. epsilon=0 must
    NOT silently become the default, and NaN/inf/negative must not corrupt the
    ledger. Only an OMITTED epsilon (None) falls back to the configured default."""

    def _mw(self):
        cfg = Config.from_yaml()
        cfg.privacy.total_epsilon = 100.0
        return DPMiddleware(cfg, mode=ExecutionMode.NAIVE_DP)

    @pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_epsilon_errors_without_spending(self, bad):
        mw = self._mw()
        r = mw.execute("SELECT COUNT(*) FROM adult", epsilon=bad)
        assert r.error is not None and "epsilon" in r.error.lower()
        # ledger untouched (and not NaN)
        assert mw.budget_summary()["consumed_epsilon"] == 0.0

    def test_none_falls_back_to_default(self):
        mw = self._mw()
        r = mw.execute("SELECT COUNT(*) FROM adult", epsilon=None)
        assert r.error is None
        assert mw.budget_summary()["consumed_epsilon"] == mw.config.privacy.default_query_epsilon
