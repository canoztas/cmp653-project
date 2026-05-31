"""Alternative distinct-template (u_k) predictors for the budget allocator.

The predictive allocator must estimate ``u_k`` --- the number of *distinct*
query templates a workload of ``k`` queries will contain --- from a short
observed prefix of ``n`` queries, in order to split a privacy budget as
``eps_q = B / u_k``. This is exactly the classic *unseen-species* extrapolation
problem: each template is a "species", each query an individual drawn from the
template-frequency distribution.

The allocator currently uses a **plug-in occupancy** estimate (Laplace-smoothed
empirical frequencies fed into ``sum_i [1-(1-p_i)^k]``). That estimate can only
ever count templates it has already *seen*, so it systematically under-predicts
``u_k`` when the prefix is short. This module adds the standard alternatives
from the unseen-species literature so they can be compared:

- ``predict_plugin``       -- current method (Laplace-smoothed occupancy).
- ``predict_good_toulmin`` -- Good & Toulmin (1956); closed-form new-species
                              extrapolation, reliable for horizon factor t<=1.
- ``predict_smoothed_gt``  -- Smoothed Good-Toulmin (Orlitsky, Suresh & Wu,
                              PNAS 2016); Poisson smoothing tames the divergent
                              GT series and stays reliable to t ~ log n.
- ``predict_chao1``        -- Chao1 (1984) richness *lower bound*; estimates the
                              asymptotic total richness, NOT the horizon-k count,
                              so it is reported only as an upper-reference.

References (verified):
  I. J. Good, Biometrika 40 (1953) 237-264.
  I. J. Good & G. H. Toulmin, Biometrika 43 (1956) 45-63.
  B. Efron & R. Thisted, Biometrika 63 (1976) 435-447.
  A. Chao, Scand. J. Statist. 11 (1984) 265-270.
  A. Orlitsky, A. T. Suresh & Y. Wu, PNAS 113 (2016) 13283-13288.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def frequency_of_frequencies(counts) -> dict[int, int]:
    """phi[r] = number of templates observed exactly r times in the prefix."""
    phi: Counter = Counter()
    for c in counts:
        if c > 0:
            phi[int(c)] += 1
    return dict(phi)


def _poisson_sf(r: int, lam: float) -> float:
    """P(Poisson(lam) >= r) = 1 - sum_{j<r} e^{-lam} lam^j / j!  (r small)."""
    if r <= 0:
        return 1.0
    if not math.isfinite(lam):
        return 1.0
    cdf = 0.0
    term = math.exp(-lam)  # j=0 term
    cdf += term
    for j in range(1, r):
        term *= lam / j
        cdf += term
    return max(0.0, 1.0 - cdf)


def predict_plugin(counts, n: int, k: int, pseudocount: float = 0.5) -> float:
    """Current allocator estimator: Laplace-smoothed empirical frequencies fed
    into the occupancy expectation E[u_k] = sum_i [1-(1-p_i)^k].

    Support is only over *observed* templates, so this can never exceed the
    number already seen --- the under-prediction this module is meant to expose.
    """
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.size == 0:
        return float(k)
    p = counts + pseudocount
    p = p / p.sum()
    return float(np.sum(1.0 - (1.0 - p) ** k))


def predict_good_toulmin(counts, n: int, k: int) -> float:
    """Good-Toulmin (1956): u_k ~= D_n + sum_{r>=1} (-1)^{r-1} t^r phi_r,
    where t = (k-n)/n is the horizon factor and D_n is the distinct count in
    the prefix. Reliable for t <= 1 (k <= 2n); the alternating series diverges
    for larger t, which is exactly what the smoothed version fixes."""
    phi = frequency_of_frequencies(counts)
    d_n = sum(phi.values())
    if n <= 0:
        return float(k)
    t = (k - n) / n
    if t <= 0:
        return float(d_n)
    new = sum(((-1) ** (r - 1)) * (t ** r) * c for r, c in phi.items())
    return float(d_n + new)


def predict_smoothed_gt(counts, n: int, k: int) -> float:
    """Smoothed Good-Toulmin (Orlitsky-Suresh-Wu, PNAS 2016), Poisson-smoothed
    variant: each term r is down-weighted by w_r = P(Poisson(r*) >= r), with the
    smoothing level r* = 1/2 * ln(n t^2 / (t-1)) chosen so the alternating-sign
    cancellation controls bias up to horizon t ~ log n. For t <= 1 the raw
    Good-Toulmin series is already convergent, so no smoothing is applied."""
    phi = frequency_of_frequencies(counts)
    d_n = sum(phi.values())
    if n <= 0:
        return float(k)
    t = (k - n) / n
    if t <= 0:
        return float(d_n)
    if t <= 1.0:
        new = sum(((-1) ** (r - 1)) * (t ** r) * c for r, c in phi.items())
        return float(d_n + new)
    r_star = max(0.0, 0.5 * math.log(n * t * t / (t - 1.0)))
    new = sum(
        ((-1) ** (r - 1)) * (t ** r) * c * _poisson_sf(r, r_star)
        for r, c in phi.items()
    )
    return float(d_n + new)


def predict_chao1(counts, n: int, k: int) -> float:
    """Chao1 (1984) richness estimator: D_n + f1^2/(2 f2) (bias-corrected when
    f2 = 0). Estimates the asymptotic TOTAL number of templates, not the count
    at horizon k, so it has no k dependence --- reported only as an
    upper-reference, never as the horizon-aware predictor."""
    phi = frequency_of_frequencies(counts)
    d_n = sum(phi.values())
    f1 = phi.get(1, 0)
    f2 = phi.get(2, 0)
    if f2 > 0:
        extra = (f1 * f1) / (2.0 * f2)
    else:
        extra = f1 * (f1 - 1) / 2.0
    return float(d_n + extra)


# Registry used by the benchmark and (optionally) the allocator.
PREDICTORS = {
    "plugin": predict_plugin,
    "good_toulmin": predict_good_toulmin,
    "smoothed_gt": predict_smoothed_gt,
    "chao1": predict_chao1,
}
