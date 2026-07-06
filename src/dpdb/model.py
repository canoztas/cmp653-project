"""Analytical model linking workload structure to DP budget consumption.

This module implements the R2 statistical model from the project's revision
plan. The model predicts:

  1. Expected unique queries E[u_k] under an arbitrary template distribution
  2. Expected privacy budget under workload-aware accounting
  3. Budget savings ratio vs naive sequential composition
  4. Concentration of u_k via McDiarmid's bounded-differences inequality
  5. Utility prediction under a fixed total budget
  6. Temporal extension with staleness tolerance / data updates

References (within the project):
- report/R2_model_sketch.md  — full derivation
- report/final_report.tex    — Section 4 "Analytical Model"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Distribution generators
# ---------------------------------------------------------------------------

def zipf_distribution(m: int, alpha: float) -> np.ndarray:
    """Zipf probability distribution over m templates with shape parameter alpha.

    p_i = i^(-alpha) / H_{m, alpha}, where H is the generalized harmonic number.
    Returns a numpy array of length m summing to 1.

    Special cases:
      alpha = 0  -> uniform distribution
      alpha -> infinity -> degenerate (mass on template 1)
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    ranks = np.arange(1, m + 1, dtype=float)
    if alpha == 0:
        weights = np.ones(m)
    else:
        weights = ranks ** (-alpha)
    return weights / weights.sum()


def uniform_distribution(m: int) -> np.ndarray:
    """Uniform distribution over m templates."""
    return np.ones(m) / m


def deterministic_distribution(m: int) -> np.ndarray:
    """All mass on the first template (perfect-repetition limit)."""
    p = np.zeros(m)
    p[0] = 1.0
    return p


# ---------------------------------------------------------------------------
# Core analytical predictions
# ---------------------------------------------------------------------------

def expected_unique_queries(p: Sequence[float], k: int) -> float:
    """Predict E[u_k] = sum_i (1 - (1 - p_i)^k).

    Proposition 1 of the R2 model: expected number of distinct templates seen
    after k i.i.d. samples from distribution {p_i}.
    """
    p_arr = np.asarray(p, dtype=float)
    # numerical safety: clamp into [0, 1]
    p_arr = np.clip(p_arr, 0.0, 1.0)
    return float(np.sum(1.0 - (1.0 - p_arr) ** k))


def expected_unique_queries_sticky(p: Sequence[float], k: int, s: float) -> float:
    """E[u_k] under a *sticky* Markov arrival process (bursty workloads).

    Arrival model: X_1 ~ p; for t>=2, X_t = X_{t-1} with probability s (repeat),
    else a fresh i.i.d. draw from p. The stationary marginal is p, so this is the
    i.i.d. model's Markovian generalisation with a single burstiness knob s.

    Closed form. The set of DISTINCT templates visited equals the set of distinct
    values among the *fresh draws* (a repeat only re-emits the current template),
    and the number of fresh draws is F = 1 + Binomial(k-1, 1-s). Conditioning on
    F and using E[z^Binomial(n,1-s)] = (s + (1-s)z)^n gives

        E[u_k] = sum_i [ 1 - (1-p_i) * (s + (1-s)(1-p_i))^{k-1} ].

    Limits: s=0 recovers the i.i.d. expected_unique_queries; s=1 gives 1 (only the
    first draw is ever seen). For 0<s<1 it is <= the i.i.d. value, which is why the
    i.i.d. forecast *over*-predicts u_k on bursty streams (the safe direction).
    """
    if not 0.0 <= s <= 1.0:
        raise ValueError("stickiness s must be in [0, 1]")
    if k <= 0:
        return 0.0
    p_arr = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    one_minus = 1.0 - p_arr
    return float(np.sum(1.0 - one_minus * (s + (1.0 - s) * one_minus) ** (k - 1)))


def expected_unique_queries_markov(P: Sequence[Sequence[float]],
                                   nu: Sequence[float], k: int) -> float:
    """E[u_k] under a GENERAL Markov arrival process with transition matrix P
    (row-stochastic, P[j][l] = Pr[next=l | current=j]) and initial distribution
    nu (X_1 ~ nu, X_{t+1} ~ P[X_t]).

    Closed form. A state i is *unvisited* in k steps iff the chain avoids i for
    all k steps. Restricting to the (m-1) states S_i = {all but i}, with Q_i the
    sub-stochastic block P[S_i, S_i] and nu_i the initial mass on S_i,
        Pr[i never visited] = nu_i^T Q_i^{k-1} 1,
    so E[u_k] = sum_i [ 1 - nu_i^T Q_i^{k-1} 1 ].

    This subsumes the earlier closed forms: with every row of P equal to nu={p_i}
    it recovers the i.i.d. expected_unique_queries; with the sticky kernel
    P[j][j]=s+(1-s)p_j, P[j][l]=(1-s)p_l (l!=j) and nu={p_i} it recovers
    expected_unique_queries_sticky. Cost O(m * (m-1)^3 log k) via matrix powers;
    practical for the moderate template universes the forecast targets.
    """
    if k <= 0:
        return 0.0
    P_arr = np.asarray(P, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    m = nu_arr.shape[0]
    if P_arr.shape != (m, m):
        raise ValueError("P must be m x m matching len(nu)")
    idx = np.arange(m)
    total = 0.0
    for i in range(m):
        keep = idx != i
        Qi = P_arr[np.ix_(keep, keep)]          # (m-1)x(m-1) sub-stochastic block
        nu_i = nu_arr[keep]
        vec = np.linalg.matrix_power(Qi, k - 1) @ np.ones(m - 1)  # Q_i^{k-1} 1
        total += 1.0 - float(nu_i @ vec)        # 1 - Pr[i never visited]
    return float(total)


def expected_unique_queries_hmm(transition: Sequence[Sequence[float]],
                                init: Sequence[float],
                                emission: Sequence[Sequence[float]],
                                k: int) -> float:
    """E[u_k] under a non-stationary, latent-state arrival process: a hidden
    Markov chain modulates which template distribution is in force.

    A hidden state h_t follows a Markov chain (row-stochastic ``transition`` of
    shape H x H, initial law ``init`` of length H); in state h the query is drawn
    from ``emission[h]`` (an H x m row-stochastic matrix over the m templates).
    This is the regime-switching / Markov-modulated workload the reviewer asks
    for: {p_i} is not fixed but changes with a latent state.

    Closed form. Template i is unvisited over k steps iff every step avoids it.
    With D_i = diag(1 - emission[:,i]) the per-state "avoid i" factors,
        Pr[i never visited] = init^T D_i (transition D_i)^{k-1} 1,
    so E[u_k] = sum_i [1 - init^T D_i (transition D_i)^{k-1} 1].

    This subsumes every earlier occupancy formula: one hidden state recovers the
    i.i.d. expected_unique_queries; a deterministic emission (hidden state = the
    template, emission = identity) recovers expected_unique_queries_markov, and
    hence the sticky and i.i.d. special cases.
    """
    if k <= 0:
        return 0.0
    P = np.asarray(transition, dtype=float)
    om = np.asarray(init, dtype=float)
    E = np.asarray(emission, dtype=float)
    H, m = E.shape
    if P.shape != (H, H) or om.shape != (H,):
        raise ValueError("transition must be HxH and init length H, matching emission rows")
    ones = np.ones(H)
    total = 0.0
    for i in range(m):
        Di = np.diag(1.0 - E[:, i])                  # per-state prob of NOT emitting i
        vec = np.linalg.matrix_power(P @ Di, k - 1) @ ones
        total += 1.0 - float(om @ Di @ vec)
    return float(total)


def expected_budget_workload_aware(p: Sequence[float], k: int, eps_q: float) -> float:
    """E[ε_wa(k)] = ε_q * E[u_k]   (Proposition 2)."""
    return eps_q * expected_unique_queries(p, k)


def expected_budget_naive(k: int, eps_q: float) -> float:
    """ε_naive(k) = k * ε_q  (textbook sequential composition)."""
    return k * eps_q


def budget_savings_ratio(p: Sequence[float], k: int) -> float:
    """S(k) = 1 - E[u_k]/k   (Proposition 2 corollary).

    Verified limits in the R2 sketch:
      - Deterministic (p_1=1):       S(k) = 1 - 1/k
      - Uniform m->infinity, fixed k: S(k) -> 0
    """
    if k <= 0:
        return 0.0
    return 1.0 - expected_unique_queries(p, k) / k


# ---------------------------------------------------------------------------
# Concentration bound (Proposition 4)
# ---------------------------------------------------------------------------

def occupancy_variance(p: Sequence[float], k: int) -> float:
    """Variance-aware upper bound on Var[u_k] (Proposition 4, tight version).

    u_k = sum_i X_i with X_i = 1[template i appears] and q_i = 1-(1-p_i)^k.
    The occupancy indicators are negatively associated, so
        Var[u_k] <= sum_i q_i (1 - q_i) = sum_i (1-(1-p_i)^k)(1-p_i)^k <= m/4.
    This is O(m), unlike McDiarmid's O(k) sub-Gaussian proxy, so it stays
    informative in the saturated regime where u_k is bounded by m. Chebyshev
    then gives P(|u_k - E[u_k]| >= t) <= occupancy_variance(p, k) / t**2.
    """
    p_arr = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    q = 1.0 - (1.0 - p_arr) ** k
    return float(np.sum(q * (1.0 - q)))


def mcdiarmid_tail_bound(k: int, deviation: float) -> float:
    """Upper bound P(|u_k - E[u_k]| >= deviation) using McDiarmid's inequality.

    Since u_k changes by at most 1 when one query changes:
      P(|u_k - E[u_k]| >= t) <= 2 * exp(-2 * t^2 / k)
    """
    if k <= 0 or deviation < 0:
        return 1.0
    return min(1.0, 2.0 * math.exp(-2.0 * deviation ** 2 / k))


def budget_exhaustion_upper_bound(p: Sequence[float], k: int, c: int) -> float:
    """P(budget_exhausted_at_step_k) where exhaustion = (u_k > c).

    Returns an upper bound via McDiarmid:
      P(u_k > c) <= exp(-2 * (c - E[u_k])^2 / k)   when c > E[u_k]
      otherwise 1.0
    """
    mu = expected_unique_queries(p, k)
    if c <= mu:
        return 1.0
    return min(1.0, math.exp(-2.0 * (c - mu) ** 2 / k))


# ---------------------------------------------------------------------------
# Utility prediction (Proposition 5)
# ---------------------------------------------------------------------------

@dataclass
class UtilityPrediction:
    """Predicted error metrics under the two strategies for a fixed budget."""
    eps_q_naive: float
    eps_q_workload_aware: float
    expected_abs_error_naive: float
    expected_abs_error_workload_aware: float
    error_ratio: float  # naive_error / wa_error  (higher = bigger win for wa)


def predict_utility_fixed_budget(
    p: Sequence[float],
    k: int,
    eps_total: float,
    sensitivity: float = 1.0,
) -> UtilityPrediction:
    """Predict per-query error under fixed total budget.

    Naive splits eps_total over k queries.
    Workload-aware splits eps_total over E[u_k] unique releases.

    Proposition 5: error_naive / error_wa = k / E[u_k].
    """
    if eps_total <= 0 or k <= 0:
        raise ValueError("eps_total and k must be positive")
    eu = expected_unique_queries(p, k)
    eu = max(eu, 1e-9)  # numerical safety
    eps_q_naive = eps_total / k
    eps_q_wa = eps_total / eu
    err_naive = sensitivity / eps_q_naive
    err_wa = sensitivity / eps_q_wa
    return UtilityPrediction(
        eps_q_naive=eps_q_naive,
        eps_q_workload_aware=eps_q_wa,
        expected_abs_error_naive=err_naive,
        expected_abs_error_workload_aware=err_wa,
        error_ratio=err_naive / err_wa if err_wa > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Temporal extension (Proposition 6 / R3)
# ---------------------------------------------------------------------------

@dataclass
class TemporalRegime:
    """Parameters of the temporal extension."""
    horizon_T: float            # time horizon (e.g., hours)
    staleness_tolerance: float  # tau: cache validity duration
    update_rate: float = 0.0    # lambda: Poisson rate of data updates (per unit time)
    update_invalidation_prob: float = 0.0  # fraction of cache invalidated per update


def expected_renoising_count(regime: TemporalRegime) -> float:
    """Expected number of noising events per cached entry over horizon T (>= 1).

    Each entry is released (noised) at least once; staleness and updates force
    additional re-noisings:
      - Staleness-forced releases over the horizon: ceil(T / tau), floored at 1 so
        the static regime (tau -> infinity) reduces to a single release, NOT zero.
      - Updates that invalidate the entry: T * lambda * invalidation_prob
    """
    forced = max(1, math.ceil(regime.horizon_T / max(regime.staleness_tolerance, 1e-9)))
    update_driven = regime.horizon_T * regime.update_rate * regime.update_invalidation_prob
    return forced + update_driven


def expected_budget_temporal(
    p: Sequence[float],
    k_total: int,
    eps_q: float,
    regime: TemporalRegime,
) -> float:
    """Expected total budget under workload-aware + temporal staleness.

    E[eps_temporal] = E[u_inf-like-cap] * N_re-noisings * eps_q

    Approximation: each distinct template seen during the horizon must be
    re-noised once per validity window. We treat E[u_k_total] as the cap
    on distinct templates seen at all.
    """
    u_total = expected_unique_queries(p, k_total)
    n_renoising = expected_renoising_count(regime)
    return eps_q * u_total * n_renoising


# ---------------------------------------------------------------------------
# Sanity tests (run as module)
# ---------------------------------------------------------------------------

def _self_check():
    """Verify the two limits from the R2 model sketch."""
    print("Self-check: limits of E[u_k] and budget savings ratio")
    print("=" * 60)

    k_values = [10, 100, 1000]

    # Limit A: deterministic (p_1 = 1)
    print("\n[Limit A] Perfect repetition (p_1 = 1, others 0):")
    print(f"{'k':>6} | {'E[u_k]':>10} | {'savings S(k)':>14} | {'1 - 1/k':>10}")
    p_det = deterministic_distribution(m=10)
    for k in k_values:
        eu = expected_unique_queries(p_det, k)
        s = budget_savings_ratio(p_det, k)
        toy = 1 - 1 / k
        print(f"{k:>6} | {eu:>10.4f} | {s:>14.4f} | {toy:>10.4f}")

    # Limit B: uniform, m large
    print("\n[Limit B] Uniform over m = 10000 templates, k fixed:")
    print(f"{'k':>6} | {'E[u_k]':>10} | {'savings S(k)':>14}")
    p_unif = uniform_distribution(m=10000)
    for k in k_values:
        eu = expected_unique_queries(p_unif, k)
        s = budget_savings_ratio(p_unif, k)
        print(f"{k:>6} | {eu:>10.4f} | {s:>14.4f}")

    # Zipf interpolation
    print("\n[Zipf] Sweep alpha for m=10, k=100:")
    print(f"{'alpha':>6} | {'E[u_k]':>10} | {'savings S(k)':>14}")
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]:
        p = zipf_distribution(m=10, alpha=alpha)
        eu = expected_unique_queries(p, 100)
        s = budget_savings_ratio(p, 100)
        print(f"{alpha:>6.2f} | {eu:>10.4f} | {s:>14.4f}")

    # Utility prediction
    print("\n[Utility] Fixed budget eps_total=10, sensitivity=1, k=100:")
    print(f"{'alpha':>6} | {'eps_q_naive':>12} | {'eps_q_wa':>10} | {'err_ratio':>10}")
    for alpha in [0.0, 1.0, 2.0]:
        p = zipf_distribution(m=10, alpha=alpha)
        pred = predict_utility_fixed_budget(p, k=100, eps_total=10.0)
        print(f"{alpha:>6.2f} | {pred.eps_q_naive:>12.4f} | "
              f"{pred.eps_q_workload_aware:>10.4f} | {pred.error_ratio:>10.4f}")

    # Temporal regime
    print("\n[Temporal] Effect of staleness tolerance tau:")
    p = zipf_distribution(m=10, alpha=1.0)
    print(f"{'tau':>6} | {'E[eps_temporal]':>16}")
    for tau in [1, 10, 100, float("inf")]:
        if tau == float("inf"):
            n = 1
        else:
            regime = TemporalRegime(horizon_T=100, staleness_tolerance=tau)
            n = expected_renoising_count(regime)
        b = expected_unique_queries(p, k=100) * n * 1.0  # eps_q=1
        print(f"{tau:>6} | {b:>16.4f}")


if __name__ == "__main__":
    _self_check()
