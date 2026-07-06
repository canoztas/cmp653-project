"""Elastic / smooth sensitivity for a single two-table equi-join COUNT.

This module implements ELASTIC SENSITIVITY (Johnson, Near & Song, "Towards
Practical Differential Privacy for SQL Queries", PVLDB 2018 -- the FLEX system)
restricted to the one case the rest of this project supports: a single inner
equi-join ``r1 JOIN r2 ON r1.A = r2.B`` whose query is ``COUNT(*)``.

It is offered as a DATA-DEPENDENT, instance-aware ALTERNATIVE to the conservative
public ``d_max`` clamp used by ``analyzer.analyze_sensitivity``. The conservative
clamp remains the default; nothing here changes it. This path is opt-in.

What is proven sound here (and tested in tests/test_join_sensitivity.py)
------------------------------------------------------------------------
Privacy unit: tuple-level DP -- add or remove ONE row from EITHER base relation
(the same unit the single-table analyzer uses).

Local sensitivity of the join COUNT at the true instance.
    Adding/removing one tuple of r1 with key value a changes the join output by
    exactly ``freq(a, r2)`` (the number of r2 rows whose B = a); symmetrically a
    tuple of r2 moves the count by ``freq(a, r1)``. The worst case over which
    tuple is added/removed is therefore

        LS(r1 |><| r2) = max( mf(A, r1), mf(B, r2) )                        (1)

    where ``mf(X, r)`` is the maximum frequency (max count of any single value)
    of the join key X in relation r. Equation (1) is exact, not a bound.

Elastic sensitivity at distance k.
    A database at Hamming distance <= k differs by at most k inserted/deleted
    tuples, and inserting k tuples raises any value's frequency by at most k, so
    ``mf'(X, r) <= mf(X, r) + k`` for every distance-k neighbour. Hence the local
    sensitivity anywhere within distance k is bounded above by

        ES(k) = max( mf(A, r1) + k, mf(B, r2) + k )
              = max( mf(A, r1), mf(B, r2) ) + k                            (2)

    ES(k) >= LS at every database within distance k. This is the elastic
    sensitivity for the single-join COUNT. (The full FLEX recursion adds a
    multiplicative product term for CHAINS of joins; with a single join the
    product term degenerates and (2) is the bound. Multi-way joins are honest
    future work -- see module note below.)

Smooth sensitivity (Nissim-Raskhodnikova-Smith 2007).
    The beta-smooth sensitivity is the smallest beta-smooth upper bound on local
    sensitivity:

        S*_beta = max_{k >= 0} e^{-k*beta} * ES(k)
                = max_{k >= 0} e^{-k*beta} * (LS0 + k)                      (3)

    with LS0 = max(mf(A,r1), mf(B,r2)). Because ES(k) grows linearly while the
    e^{-k*beta} weight decays, the maximiser is a small finite k* and (3) is
    computed exactly by scanning k.

(eps, delta) relaxation -- stated honestly.
    Smooth sensitivity does NOT give pure eps-DP for additive noise here. The
    NRS framework releases ``f + S*_beta/alpha * Z`` and gives
    (eps, delta)-DP, NOT (eps, 0)-DP, for the calibrations used below:

      * Laplace Z with beta = eps / (2*ln(2/delta)) and scale lambda =
        2*S*_beta / eps   -> (eps, delta)-DP.
      * Gaussian Z with beta = eps / (2*(C)) ...; we use the standard
        analytic-Gaussian-style calibration alpha = eps / (2),
        beta = eps / (2 * ln(2/delta)), scale sigma derived from S*_beta.

    We therefore report an (eps, delta)-DP noise SCALE, and we are explicit that
    pure eps-DP is NOT achieved by this path. The default d_max clamp, by
    contrast, yields pure eps-DP global sensitivity. This is the price of the
    data-dependent tightening.

Scope / honest limitations
---------------------------
  * Single inner equi-join, COUNT(*) only. No SUM/AVG, no >2 tables, no self
    joins, no outer/cross joins. These match the parser's existing envelope.
  * The bound (2) is the conservative-but-PROVABLY-SOUND single-join elastic
    sensitivity. The tighter multiplicative FLEX form only matters for chains of
    joins, which are out of scope, so nothing is lost for the supported query.
  * mf statistics are computed from the REAL tables via DuckDB. The max
    frequencies themselves are data-dependent and are released only through the
    smooth-sensitivity-calibrated noise; we do not publish raw mf here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class JoinSensitivity:
    """Elastic / smooth sensitivity for one equi-join COUNT.

    Attributes
    ----------
    mf_left, mf_right:
        Max frequency of the join key in the left/right relation.
    local_sensitivity:
        LS0 = max(mf_left, mf_right): the exact local sensitivity (1).
    elastic_sensitivity:
        ES(0) == local_sensitivity. Kept explicit for clarity; ES(k) is
        available via :meth:`es_at`.
    smooth_sensitivity:
        S*_beta from (3) at the configured beta.
    beta:
        The smoothing parameter actually used.
    k_star:
        The distance k that attains the smooth-sensitivity maximum.
    """

    mf_left: int
    mf_right: int
    local_sensitivity: float
    elastic_sensitivity: float
    smooth_sensitivity: float
    beta: float
    k_star: int
    left_table: str = ""
    right_table: str = ""

    def es_at(self, k: int) -> float:
        """Elastic sensitivity at distance k: max(mf_left, mf_right) + k -- eq (2)."""
        if k < 0:
            raise ValueError("distance k must be >= 0")
        return self.local_sensitivity + k


def elastic_sensitivity(mf_left: int, mf_right: int) -> float:
    """Elastic sensitivity ES(0) of a single equi-join COUNT = local sensitivity.

    ES(0) = max(mf_left, mf_right)  (eq (1)/(2) at k=0). This is the exact local
    sensitivity of the join COUNT at the given max-frequency statistics.
    """
    if mf_left < 0 or mf_right < 0:
        raise ValueError("max frequencies must be non-negative")
    return float(max(mf_left, mf_right))


def es_at(mf_left: int, mf_right: int, k: int) -> float:
    """Elastic sensitivity at distance k: max(mf_left, mf_right) + k -- eq (2)."""
    if k < 0:
        raise ValueError("distance k must be >= 0")
    return elastic_sensitivity(mf_left, mf_right) + k


def smooth_sensitivity(mf_left: int, mf_right: int, beta: float) -> tuple[float, int]:
    """beta-smooth sensitivity S*_beta = max_k e^{-k*beta} (LS0 + k) -- eq (3).

    Returns (S*_beta, k_star). For ES(k) = LS0 + k the weighted value
    g(k) = e^{-k*beta}(LS0 + k) is unimodal in k, so we scan k upward and stop
    once g starts decreasing (after passing k=0). A hard cap guards beta -> 0.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0 for a finite smooth sensitivity")
    ls0 = elastic_sensitivity(mf_left, mf_right)
    best_val = ls0  # k = 0
    best_k = 0
    prev = best_val
    # Unimodal: the continuous maximiser is k* = 1/beta - LS0; cap the scan well
    # beyond it to be safe, but break as soon as g decreases (after k>=1).
    k_cap = int(max(1.0 / beta, 1)) + 64
    for k in range(1, k_cap + 1):
        val = math.exp(-k * beta) * (ls0 + k)
        if val > best_val:
            best_val = val
            best_k = k
        if val < prev and k >= 1:
            # past the unimodal peak; values only decrease from here
            break
        prev = val
    return best_val, best_k


# ---------------------------------------------------------------------------
# Noise calibration from smooth sensitivity (NRS 2007) -- (eps, delta)-DP only.
# ---------------------------------------------------------------------------

def laplace_scale_from_smooth(smooth: float, epsilon: float, delta: float) -> float:
    """Laplace noise scale for (eps, delta)-DP via smooth sensitivity (NRS 2007).

    Releasing f + (2 * S*_beta / eps) * Lap(1) with
    beta = eps / (2 * ln(2/delta)) gives (eps, delta)-DP. This is NOT (eps,0)-DP.
    Caller must have computed S*_beta with the SAME beta (see
    :func:`recommended_beta`).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return 2.0 * smooth / epsilon


def recommended_beta(epsilon: float, delta: float) -> float:
    """beta = eps / (2 ln(2/delta)) -- the NRS Laplace calibration for (eps,delta)-DP."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return epsilon / (2.0 * math.log(2.0 / delta))


def compute_join_sensitivity(
    mf_left: int,
    mf_right: int,
    epsilon: float,
    delta: float,
    left_table: str = "",
    right_table: str = "",
    beta: Optional[float] = None,
) -> JoinSensitivity:
    """Bundle ES(0) and S*_beta for a single equi-join COUNT.

    If ``beta`` is None it is set to the NRS Laplace calibration
    ``recommended_beta(epsilon, delta)`` so that ``laplace_scale_from_smooth``
    applied to the returned ``smooth_sensitivity`` is a valid (eps, delta)-DP
    noise scale.
    """
    if beta is None:
        beta = recommended_beta(epsilon, delta)
    ls0 = elastic_sensitivity(mf_left, mf_right)
    smooth, k_star = smooth_sensitivity(mf_left, mf_right, beta)
    return JoinSensitivity(
        mf_left=int(mf_left),
        mf_right=int(mf_right),
        local_sensitivity=ls0,
        elastic_sensitivity=ls0,  # ES(0)
        smooth_sensitivity=smooth,
        beta=beta,
        k_star=k_star,
        left_table=left_table,
        right_table=right_table,
    )


# ---------------------------------------------------------------------------
# Max-frequency statistics from the real tables (DuckDB).
# ---------------------------------------------------------------------------

def max_frequency(con, table: str, key: str) -> int:
    """Max frequency of ``key`` in ``table``: max over values of COUNT(*) GROUP BY key.

    ``con`` is a DuckDB connection. NULL keys do not join under an inner equi-join
    and are excluded, matching the join semantics. Returns 0 for an empty/all-NULL
    column.
    """
    row = con.execute(
        f"SELECT max(c) FROM (SELECT count(*) AS c FROM {table} "
        f"WHERE {key} IS NOT NULL GROUP BY {key})"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def join_count_sensitivity_from_db(
    con,
    left_table: str,
    left_key: str,
    right_table: str,
    right_key: str,
    epsilon: float,
    delta: float,
    beta: Optional[float] = None,
) -> JoinSensitivity:
    """End-to-end: read mf statistics from DuckDB and return the join sensitivity."""
    mf_l = max_frequency(con, left_table, left_key)
    mf_r = max_frequency(con, right_table, right_key)
    return compute_join_sensitivity(
        mf_l, mf_r, epsilon, delta,
        left_table=left_table, right_table=right_table, beta=beta,
    )
