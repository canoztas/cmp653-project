"""Tests for the online Workload Learning Layer (S2).

Two things must hold: (1) the learner is deterministic given a deterministic
input stream (no hidden randomness), and (2) it *reacts to a distribution shift*
-- after the workload switches from template set A to a disjoint set B, the
learner's estimated distribution re-centres on B and its forecast tracks the new
regime better than a static full-history plug-in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dpdb.model import zipf_distribution
from dpdb.workload_learning import (
    Forecast,
    StaticPluginBaseline,
    WorkloadLearner,
)


def _make_shift_stream(seed=7, n_half=600, m_set=30, alpha=1.1):
    """Disjoint set A (ids 0..m-1) then set B (ids m..2m-1), Zipf each half."""
    rng = np.random.default_rng(seed)
    p = zipf_distribution(m_set, alpha)
    a = rng.choice(m_set, size=n_half, p=p)
    b = rng.choice(m_set, size=n_half, p=p) + m_set
    return np.concatenate([a, b]).tolist(), m_set, n_half


class TestDeterminism:
    def test_same_stream_same_forecast(self):
        stream, _, _ = _make_shift_stream(seed=1)
        la = WorkloadLearner(mode="ewma", decay=0.98)
        lb = WorkloadLearner(mode="ewma", decay=0.98)
        for tid in stream:
            la.update(tid)
            lb.update(tid)
        fa = la.forecast(100)
        fb = lb.forecast(100)
        assert fa == fb
        assert fa.plugin_uk == fb.plugin_uk
        assert fa.smoothed_gt_uk == fb.smoothed_gt_uk

    def test_no_hidden_randomness_across_instances(self):
        stream, _, _ = _make_shift_stream(seed=2)
        preds = []
        for _ in range(3):
            l = WorkloadLearner(mode="ewma", decay=0.95)
            l.update_many(stream)
            preds.append(l.expected_remaining_distinct(50))
        assert preds[0] == preds[1] == preds[2]

    def test_window_mode_deterministic(self):
        stream, _, _ = _make_shift_stream(seed=3)
        l1 = WorkloadLearner(mode="window", window=150)
        l2 = WorkloadLearner(mode="window", window=150)
        l1.update_many(stream)
        l2.update_many(stream)
        assert l1.forecast(80) == l2.forecast(80)


class TestForecastShape:
    def test_forecast_returns_dataclass(self):
        l = WorkloadLearner()
        l.update_many([0, 1, 2, 0, 1, 0])
        f = l.forecast(10)
        assert isinstance(f, Forecast)
        assert f.n_seen == 6
        assert f.horizon_k == 10
        assert f.plugin_uk >= 0

    def test_empty_learner_forecasts_zero(self):
        l = WorkloadLearner()
        f = l.forecast(100)
        assert f.plugin_uk == 0.0 and f.active_templates == 0

    def test_forecast_monotone_in_horizon(self):
        # More future queries cannot reveal fewer distinct templates (plug-in).
        l = WorkloadLearner(mode="window", window=500)
        rng = np.random.default_rng(11)
        p = zipf_distribution(20, 1.0)
        l.update_many(rng.choice(20, size=400, p=p).tolist())
        assert l.expected_remaining_distinct(200) >= l.expected_remaining_distinct(20)

    def test_negative_horizon_rejected(self):
        l = WorkloadLearner()
        l.update(0)
        try:
            l.forecast(-1)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestDriftReaction:
    def test_distribution_recenters_on_set_b(self):
        """After the shift, the learner's mass concentrates on set-B ids."""
        stream, m_set, n_half = _make_shift_stream(seed=5)
        l = WorkloadLearner(mode="ewma", decay=0.97)
        l.update_many(stream)
        # Inspect raw decayed counts: total mass on set B should dominate set A.
        counts = l._raw_counts()
        mass_a = sum(v for k, v in counts.items() if k < m_set)
        mass_b = sum(v for k, v in counts.items() if k >= m_set)
        assert mass_b > 5 * mass_a, (mass_a, mass_b)

    def test_active_templates_tracks_shift(self):
        """Mid-stream the active set is A; far past the shift it is B. The
        *count* stays near the regime size, it does not keep growing like the
        static baseline's support does."""
        stream, m_set, n_half = _make_shift_stream(seed=6)
        l = WorkloadLearner(mode="ewma", decay=0.97, active_threshold=0.002)
        # feed first half
        l.update_many(stream[:n_half])
        active_pre = l.active_templates()
        # feed second half
        l.update_many(stream[n_half:])
        active_post = l.active_templates()
        # Active count stays bounded near a single regime, not 2*m_set.
        assert active_pre <= m_set + 2
        assert active_post <= m_set + 2

    def test_drift_aware_beats_static_post_shift(self):
        """Post-shift, the EWMA forecast is closer to the realized distinct
        count of the *current* regime than the static full-history plug-in."""
        seed = 9
        stream, m_set, n_half = _make_shift_stream(seed=seed)
        horizon = 80

        learner = WorkloadLearner(mode="ewma", decay=0.99)
        static = StaticPluginBaseline()
        learner.update_many(stream)
        static.update_many(stream)

        # Realized distinct over next `horizon` draws from the post-shift (B) regime.
        rng = np.random.default_rng(seed + 100)
        p = zipf_distribution(m_set, 1.1)
        trials = 300
        ds = []
        for _ in range(trials):
            ids = rng.choice(m_set, size=horizon, p=p)
            ds.append(len(set(ids.tolist())))
        realized = float(np.mean(ds))

        drift_pred = learner.expected_remaining_distinct(horizon)
        static_pred = static.forecast_distinct(horizon)

        drift_err = abs(drift_pred - realized)
        static_err = abs(static_pred - realized)
        assert drift_err < static_err, (drift_pred, static_pred, realized)

    def test_static_support_grows_unbounded(self):
        """Sanity: the static baseline's support spans BOTH regimes (this is the
        weakness the drift-aware learner fixes)."""
        stream, m_set, _ = _make_shift_stream(seed=4)
        static = StaticPluginBaseline()
        static.update_many(stream)
        # Both sets actually appear, so distinct should exceed one regime size.
        assert static.distinct() > m_set


class TestEwmaForgetting:
    def test_effective_n_approaches_steady_state(self):
        decay = 0.99
        l = WorkloadLearner(mode="ewma", decay=decay)
        l.update_many([0] * 2000)
        steady = 1.0 / (1.0 - decay)
        assert abs(l.effective_n() - steady) <= 2.0

    def test_old_template_fades(self):
        l = WorkloadLearner(mode="ewma", decay=0.9)
        l.update(0)  # rare old template
        l.update_many([1] * 200)  # flood with a new one
        p = l.current_distribution()
        # The flood template dominates; the old one is negligible.
        assert p[0] > 0.99


class TestWindowMode:
    def test_window_counts_exact(self):
        l = WorkloadLearner(mode="window", window=3)
        l.update_many([0, 1, 2, 3])  # 0 should have been evicted
        counts = l._raw_counts()
        assert 0 not in counts
        assert set(counts.keys()) == {1, 2, 3}
        assert l.effective_n() == 3


class TestSafetyLimit:
    def test_active_templates_is_not_a_hard_cap(self):
        """Document the honest limit: active_templates is a soft active-set
        estimate, NOT m's hard cap. An adversary issuing many fresh templates in
        the window drives it up to the number seen -- it provides no upper-bound
        guarantee on future distinct templates, so B/m safety must still use the
        public m, not this number."""
        l = WorkloadLearner(mode="window", window=1000)
        # 500 distinct one-shot templates in the window.
        l.update_many(list(range(500)))
        # active count just reflects what was seen; it is not a guaranteed cap.
        assert l.active_templates() == 500
        # A subsequent fresh template was never bounded by the estimate.
        l.update(99999)
        assert 99999 in l._raw_counts()
