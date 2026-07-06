"""Tests for the R2 analytical model and R3 temporal extension."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.model import (
    budget_savings_ratio,
    deterministic_distribution,
    expected_renoising_count,
    expected_budget_temporal,
    expected_budget_workload_aware,
    expected_unique_queries,
    expected_unique_queries_sticky,
    expected_unique_queries_markov,
    expected_unique_queries_hmm,
    mcdiarmid_tail_bound,
    occupancy_variance,
    predict_utility_fixed_budget,
    TemporalRegime,
    uniform_distribution,
    zipf_distribution,
)


class TestZipfDistribution:
    def test_sums_to_one(self):
        for m in [5, 10, 100]:
            for alpha in [0.0, 0.5, 1.0, 2.0]:
                p = zipf_distribution(m, alpha)
                assert abs(p.sum() - 1.0) < 1e-9

    def test_alpha_zero_is_uniform(self):
        p = zipf_distribution(10, alpha=0.0)
        assert np.allclose(p, 1.0 / 10)

    def test_high_alpha_concentrates(self):
        p = zipf_distribution(10, alpha=5.0)
        assert p[0] > 0.9  # heavy concentration on rank-1


class TestOccupancyVariance:
    def test_bounded_by_m_over_4_and_positive(self):
        # Var[u_k] <= m/4 (Prop 4 variance-aware bound), and positive when not saturated
        from dpdb.model import zipf_distribution
        p = zipf_distribution(10, 1.0)
        V = occupancy_variance(p, 100)
        assert 0.0 < V <= 10 / 4 + 1e-9

    def test_matches_monte_carlo_std(self):
        # sqrt(V) tracks the empirical std of u_k (the whole point of the fix)
        from dpdb.model import zipf_distribution
        p = zipf_distribution(10, 1.0)
        rng = np.random.default_rng(0)
        uk = [len(set(rng.choice(10, size=100, p=p).tolist())) for _ in range(4000)]
        assert occupancy_variance(p, 100) ** 0.5 == pytest.approx(float(np.std(uk)), abs=0.05)

    def test_far_tighter_than_mcdiarmid_under_saturation(self):
        # the saturation regime: variance-aware std << McDiarmid sub-Gaussian scale sqrt(k)/2
        from dpdb.model import zipf_distribution
        p = zipf_distribution(10, 1.0)
        assert occupancy_variance(p, 100) ** 0.5 < (100 ** 0.5) / 2 / 5  # >5x tighter


class TestExpectedUniqueQueries:
    def test_deterministic_limit(self):
        """Limit A: perfect repetition gives E[u_k] = 1."""
        p = deterministic_distribution(m=10)
        for k in [1, 10, 100, 1000]:
            assert expected_unique_queries(p, k) == pytest.approx(1.0, abs=1e-9)

    def test_uniform_limit(self):
        """Limit B: uniform with m large gives E[u_k] ≈ k for k << m."""
        p = uniform_distribution(m=10000)
        for k in [10, 50, 100]:
            eu = expected_unique_queries(p, k)
            # E[u_k] = m * (1 - (1 - 1/m)^k) ≈ k for k << m
            assert abs(eu - k) < 1.0  # within 1 unit

    def test_truncated_uniform(self):
        """Limit C: uniform with m small, k large: E[u_k] -> m."""
        p = uniform_distribution(m=5)
        eu = expected_unique_queries(p, k=10000)
        assert abs(eu - 5.0) < 1e-3


class TestStickyMarkovOccupancy:
    """Closed-form occupancy under the sticky Markov arrival process:
    E[u_k] = sum_i [1 - (1-p_i)(s + (1-s)(1-p_i))^{k-1}]."""

    def test_s0_recovers_iid(self):
        p = zipf_distribution(20, 1.0)
        for k in (10, 100, 200):
            assert (expected_unique_queries_sticky(p, k, 0.0)
                    == pytest.approx(expected_unique_queries(p, k), abs=1e-9))

    def test_s1_gives_single_template(self):
        # always repeat the first draw -> exactly one distinct template
        p = zipf_distribution(20, 1.0)
        for k in (5, 50, 500):
            assert expected_unique_queries_sticky(p, k, 1.0) == pytest.approx(1.0, abs=1e-9)

    def test_monotone_decreasing_in_s(self):
        # more burstiness -> fewer distinct templates
        p = zipf_distribution(30, 1.0)
        vals = [expected_unique_queries_sticky(p, 100, s) for s in (0.0, 0.3, 0.6, 0.9)]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def test_matches_simulation(self):
        p = zipf_distribution(25, 1.0)
        k, s = 80, 0.6
        closed = expected_unique_queries_sticky(p, k, s)
        rng = np.random.default_rng(123)
        sims = []
        for _ in range(3000):
            seq = np.empty(k, dtype=int)
            seq[0] = rng.choice(len(p), p=p)
            for t in range(1, k):
                seq[t] = seq[t - 1] if rng.random() < s else rng.choice(len(p), p=p)
            sims.append(len(set(seq.tolist())))
        assert closed == pytest.approx(float(np.mean(sims)), abs=0.3)


class TestGeneralMarkovOccupancy:
    """General-Markov occupancy E[u_k] = sum_i [1 - nu_i^T Q_i^{k-1} 1], which
    subsumes the i.i.d. and sticky closed forms."""

    def test_recovers_iid(self):
        p = zipf_distribution(8, 1.0)
        P = np.tile(p, (8, 1))                  # every row = p  -> i.i.d.
        for k in (10, 50, 120):
            assert (expected_unique_queries_markov(P, p, k)
                    == pytest.approx(expected_unique_queries(p, k), abs=1e-9))

    def test_recovers_sticky(self):
        p = zipf_distribution(10, 0.8)
        s = 0.6
        P = (1 - s) * np.tile(p, (10, 1)) + s * np.eye(10)
        for k in (10, 80):
            assert (expected_unique_queries_markov(P, p, k)
                    == pytest.approx(expected_unique_queries_sticky(p, k, s), abs=1e-9))

    def test_matches_simulation_general_chain(self):
        rng = np.random.default_rng(0)
        m, k = 6, 60
        M = rng.random((m, m)) + 0.05
        P = M / M.sum(1, keepdims=True)
        nu = rng.random(m); nu /= nu.sum()
        closed = expected_unique_queries_markov(P, nu, k)
        sims = []
        for t in range(3000):
            r = np.random.default_rng(5000 + t)
            x = r.choice(m, p=nu); seen = {x}
            for _ in range(k - 1):
                x = r.choice(m, p=P[x]); seen.add(x)
            sims.append(len(seen))
        assert closed == pytest.approx(float(np.mean(sims)), abs=0.2)


class TestMarkovModulatedOccupancy:
    """Latent-state (HMM) occupancy E[u_k]=sum_i[1-omega^T D_i (T D_i)^{k-1} 1],
    which subsumes the i.i.d. and general-Markov forms."""

    def test_single_state_recovers_iid(self):
        p = zipf_distribution(8, 1.0)
        T = np.array([[1.0]]); om = np.array([1.0]); E = p.reshape(1, -1)
        for k in (10, 60, 150):
            assert (expected_unique_queries_hmm(T, om, E, k)
                    == pytest.approx(expected_unique_queries(p, k), abs=1e-9))

    def test_deterministic_emission_recovers_general_markov(self):
        rng = np.random.default_rng(3); m = 6
        Mx = rng.random((m, m)) + 0.1; P = Mx / Mx.sum(1, keepdims=True)
        nu = rng.random(m); nu /= nu.sum()
        for k in (20, 60):
            assert (expected_unique_queries_hmm(P, nu, np.eye(m), k)
                    == pytest.approx(expected_unique_queries_markov(P, nu, k), abs=1e-9))

    def test_matches_simulation_regime_switching(self):
        m = 12; half = m // 2
        eA = np.zeros(m); eA[:half] = 1.0 / half
        eB = np.zeros(m); eB[half:] = 1.0 / (m - half)
        E = np.vstack([eA, eB])
        T = np.array([[0.9, 0.1], [0.1, 0.9]]); om = np.array([0.5, 0.5])
        k = 40
        closed = expected_unique_queries_hmm(T, om, E, k)
        sims = []
        for t in range(3000):
            rng = np.random.default_rng(11 + t)
            h = int(rng.choice(2, p=om)); seen = set()
            for _ in range(k):
                seen.add(int(rng.choice(m, p=E[h]))); h = int(rng.choice(2, p=T[h]))
            sims.append(len(seen))
        assert closed == pytest.approx(float(np.mean(sims)), abs=0.3)


class TestBudgetSavingsRatio:
    def test_perfect_repetition_recovers_one_minus_one_over_k(self):
        """The toy 1 - 1/k formula is the corner case of perfect repetition."""
        p = deterministic_distribution(m=10)
        for k in [10, 100, 1000]:
            s = budget_savings_ratio(p, k)
            assert s == pytest.approx(1.0 - 1.0 / k, abs=1e-9)

    def test_uniform_large_m_gives_zero_savings(self):
        """Naive limit: uniform over many templates gives no savings."""
        p = uniform_distribution(m=10000)
        s = budget_savings_ratio(p, k=100)
        assert s < 0.01

    def test_zipf_intermediate(self):
        """Zipf interpolates between the two limits."""
        for alpha, expected_savings_lower in [(0.0, 0.0), (1.0, 0.5), (3.0, 0.7)]:
            p = zipf_distribution(m=10, alpha=alpha)
            s = budget_savings_ratio(p, k=100)
            assert s >= expected_savings_lower or alpha == 0.0


class TestUtilityPrediction:
    def test_naive_vs_workload_aware_under_fixed_budget(self):
        p = zipf_distribution(m=10, alpha=2.0)
        pred = predict_utility_fixed_budget(p, k=100, eps_total=10.0, sensitivity=1.0)
        # workload-aware should give more per-query budget -> less error
        assert pred.eps_q_workload_aware > pred.eps_q_naive
        assert pred.expected_abs_error_workload_aware < pred.expected_abs_error_naive
        assert pred.error_ratio > 1.0


class TestConcentration:
    def test_tail_bound_is_probability(self):
        for k in [10, 100, 1000]:
            for t in [0, 1, 5, 10]:
                b = mcdiarmid_tail_bound(k, t)
                assert 0.0 <= b <= 1.0

    def test_tail_bound_decreases_with_t(self):
        b1 = mcdiarmid_tail_bound(100, 1)
        b2 = mcdiarmid_tail_bound(100, 10)
        assert b2 < b1


class TestTemporalExtension:
    def test_renoising_count_increases_with_shorter_tau(self):
        r_long = TemporalRegime(horizon_T=100, staleness_tolerance=100)
        r_short = TemporalRegime(horizon_T=100, staleness_tolerance=10)
        assert expected_renoising_count(r_short) > expected_renoising_count(r_long)

    def test_renoising_with_updates(self):
        r_no_updates = TemporalRegime(horizon_T=100, staleness_tolerance=50,
                                      update_rate=0.0)
        r_with_updates = TemporalRegime(horizon_T=100, staleness_tolerance=50,
                                        update_rate=0.5, update_invalidation_prob=0.2)
        assert expected_renoising_count(r_with_updates) > expected_renoising_count(r_no_updates)

    def test_temporal_budget_recovers_static_in_limit(self):
        """As tau -> infinity, temporal budget reduces to static workload-aware."""
        p = zipf_distribution(m=10, alpha=1.0)
        static_budget = expected_budget_workload_aware(p, k=100, eps_q=1.0)
        regime_static = TemporalRegime(horizon_T=100, staleness_tolerance=1e9)
        temporal_budget = expected_budget_temporal(p, k_total=100, eps_q=1.0, regime=regime_static)
        # With tau >> T, only one re-noising is needed
        assert temporal_budget == pytest.approx(static_budget, rel=0.01)
