"""Markov-modulated (latent state-space) occupancy: forecasting under a
non-stationary workload whose query distribution {p_i} switches with a hidden
regime. Answers the reviewer's "advanced arrival modeling" direction by
extending the occupancy framework from a fixed (or single-Markov) {p_i} to a
hidden-state regime-switching process.

A hidden Markov chain (transition T, initial omega) modulates which template
distribution is active; in regime h the query is drawn from emission[h]. The
closed form is
    E[u_k] = sum_i [1 - omega^T D_i (T D_i)^{k-1} 1],  D_i = diag(1 - emission[:,i]),
which subsumes the i.i.d., sticky, and general-Markov forms.

We build a two-regime workload: regime A concentrates on the first half of the
templates, regime B on the second half, with slow (sticky) switching, so the
realized {p_i} is genuinely non-stationary. We check (i) the closed form matches
simulation, and (ii) a naive forecast that ignores the regimes -- feeding the
stationary marginal into the i.i.d. occupancy -- MIS-predicts u_k, which is
exactly why the latent-state model is needed.

Deterministic seeds. Run: python experiments/markov_modulated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import (expected_unique_queries, expected_unique_queries_hmm,
                        expected_unique_queries_markov, zipf_distribution)

M = 20
TRIALS = 5000


def two_regime(m, switch_stay=0.95):
    """Two regimes splitting the m templates; slow switching (sticky hidden chain)."""
    half = m // 2
    eA = np.zeros(m); eA[:half] = zipf_distribution(half, 1.0); eA /= eA.sum()
    eB = np.zeros(m); eB[half:] = zipf_distribution(m - half, 1.0); eB /= eB.sum()
    emission = np.vstack([eA, eB])
    T = np.array([[switch_stay, 1 - switch_stay], [1 - switch_stay, switch_stay]])
    omega = np.array([0.5, 0.5])
    return T, omega, emission


def _sim(T, omega, emission, k, trials, seed0):
    H, m = emission.shape
    vals = []
    for t in range(trials):
        rng = np.random.default_rng(seed0 + 7 * t)
        h = int(rng.choice(H, p=omega)); seen = set()
        for _ in range(k):
            seen.add(int(rng.choice(m, p=emission[h])))
            h = int(rng.choice(H, p=T[h]))
        vals.append(len(seen))
    return float(np.mean(vals))


def main():
    T, omega, emission = two_regime(M)
    # stationary marginal of the hidden chain (here uniform by symmetry)
    pstat = omega @ emission

    print("=== Markov-modulated (latent regime) occupancy vs simulation ===")
    print(f"    two regimes over m={M} templates, sticky hidden chain\n")
    print(f"  {'k':>4} | {'HMM closed':>10} {'sim':>8} {'|err|':>6} | "
          f"{'naive i.i.d.':>12} {'naive err':>9}")
    rows = []
    for k in (20, 40, 80, 160):
        hmm = expected_unique_queries_hmm(T, omega, emission, k)
        sim = _sim(T, omega, emission, k, TRIALS, 4000)
        naive = expected_unique_queries(pstat, k)        # ignores the regime structure
        rows.append(dict(k=k, hmm=hmm, sim=sim, abs_err=abs(hmm - sim),
                         naive_iid=naive, naive_err=abs(naive - sim)))
        print(f"  {k:>4} | {hmm:10.3f} {sim:8.3f} {abs(hmm - sim):6.3f} | "
              f"{naive:12.3f} {abs(naive - sim):9.3f}")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "markov"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "markov_modulated.csv", index=False)

    # special-case recovery (machine precision)
    p = zipf_distribution(8, 1.0)
    iid_gap = abs(expected_unique_queries_hmm(np.array([[1.0]]), np.array([1.0]),
                                              p.reshape(1, -1), 60)
                  - expected_unique_queries(p, 60))
    rng = np.random.default_rng(3); mm = 6
    Mx = rng.random((mm, mm)) + 0.1; P = Mx / Mx.sum(1, keepdims=True)
    nu = rng.random(mm); nu /= nu.sum()
    mk_gap = abs(expected_unique_queries_hmm(P, nu, np.eye(mm), 40)
                 - expected_unique_queries_markov(P, nu, 40))

    print("\n=== Headline ===")
    print(f"  Closed form matches simulation to max |err| {df.abs_err.max():.3f}; "
          f"recovers i.i.d. to {iid_gap:.1e} and general-Markov to {mk_gap:.1e}.")
    print(f"  Ignoring the regimes (naive i.i.d. on the stationary marginal) "
          f"mis-predicts u_k by up to {df.naive_err.max():.2f}")
    print(f"    (vs {df.abs_err.max():.3f} for the latent-state forecast): the "
          f"regime structure must be modelled, and now is.")
    print(f"\n  Wrote {out / 'markov_modulated.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
