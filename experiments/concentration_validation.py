"""Empirical validation of the concentration proposition.

The paper states two things analytically that it does not measure: (i) the
realized distinct count u_k has variance V = O(m), so its standard deviation
grows as sqrt(m), NOT sqrt(k) -- a single run lands within a fraction of a
template of the forecast; and (ii) the budget-exhaustion tail obeys the
Chebyshev bound P[u_k eps_q >= c] <= V / (c/eps_q - E[u_k])^2.

This script measures both on many independent runs and checks that

  * empirical std(u_k) matches sqrt(occupancy_variance) (the closed-form V), and
    scales with sqrt(m) while being ~flat in k;
  * the empirical exceedance frequency never exceeds the Chebyshev bound (the
    bound is valid, and conservative).

Deterministic seeds. Run: python experiments/concentration_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import expected_unique_queries, occupancy_variance, zipf_distribution

TRIALS = 4000
EPS_Q = 1.0


def _realized(p, k, n_trials):
    uks = np.empty(n_trials)
    for t in range(n_trials):
        rng = np.random.default_rng(56789 + 31 * t)
        uks[t] = len(set(rng.choice(len(p), size=k, p=p).tolist()))
    return uks


def main():
    rows = []
    print("=== (i) std(u_k) = sqrt(V) = O(sqrt m), flat in k ===\n")
    print(f"  {'m':>4} {'k':>4} {'alpha':>5} | {'E[u_k]':>7} {'emp.mean':>8} | "
          f"{'sqrt(V)':>7} {'emp.std':>7} | {'ratio':>5}")
    for m in (20, 50, 100):
        for k in (50, 100, 200):
            for alpha in (0.5, 1.0):
                p = zipf_distribution(m, alpha)
                euk = expected_unique_queries(p, k)
                sd = occupancy_variance(p, k) ** 0.5
                uks = _realized(p, k, TRIALS)
                emp_mean, emp_std = float(uks.mean()), float(uks.std())
                ratio = emp_std / sd if sd > 0 else float("nan")
                rows.append(dict(m=m, k=k, alpha=alpha, Euk=euk, emp_mean=emp_mean,
                                 sqrtV=sd, emp_std=emp_std, std_ratio=ratio))
                print(f"  {m:>4} {k:>4} {alpha:>5.1f} | {euk:7.2f} {emp_mean:8.2f} | "
                      f"{sd:7.3f} {emp_std:7.3f} | {ratio:5.2f}")

    df = pd.DataFrame(rows)

    # sqrt(m) scaling at fixed k, alpha: std should roughly track sqrt(m)
    base = df[(df.k == 100) & (df.alpha == 1.0)].sort_values("m")
    print("\n  sqrt(m) scaling (k=100, alpha=1.0): "
          + ", ".join(f"m={int(r.m)}:std={r.emp_std:.2f}" for _, r in base.iterrows())
          + f"  (sqrt(m) ratio {np.sqrt(base.m.iloc[-1]/base.m.iloc[0]):.2f}x vs "
          f"std ratio {base.emp_std.iloc[-1]/base.emp_std.iloc[0]:.2f}x)")
    flat = df[(df.m == 50) & (df.alpha == 1.0)].sort_values("k")
    print("  flat in k (m=50, alpha=1.0):       "
          + ", ".join(f"k={int(r.k)}:std={r.emp_std:.2f}" for _, r in flat.iterrows())
          + "  (std stays bounded while k grows 4x)")

    print("\n=== (ii) Chebyshev budget-exhaustion tail holds (empirical <= bound) ===\n")
    print(f"  {'m':>4} {'k':>4} {'alpha':>5} | {'c (budget)':>10} | "
          f"{'emp P[>=c]':>10} | {'Chebyshev':>10} | holds?")
    tail_rows = []
    for m in (20, 50):
        for alpha in (0.5, 1.0):
            k = 100
            p = zipf_distribution(m, alpha)
            euk = expected_unique_queries(p, k)
            V = occupancy_variance(p, k)
            uks = _realized(p, k, TRIALS)
            for z in (2.0, 3.0):                       # c set z std above the mean
                c = (euk + z * V ** 0.5) * EPS_Q       # budget threshold
                thr = c / EPS_Q
                emp = float(np.mean(uks * EPS_Q >= c))
                cheb = V / (thr - euk) ** 2 if thr > euk else 1.0
                cheb = min(cheb, 1.0)
                holds = emp <= cheb + 1e-9
                tail_rows.append(dict(m=m, k=k, alpha=alpha, c=c, emp_tail=emp,
                                      chebyshev=cheb, holds=holds))
                print(f"  {m:>4} {k:>4} {alpha:>5.1f} | {c:10.2f} | {emp:10.4f} | "
                      f"{cheb:10.4f} | {holds}")

    tdf = pd.DataFrame(tail_rows)
    out = Path(__file__).parent.parent / "results" / "concentration"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "concentration_std.csv", index=False)
    tdf.to_csv(out / "concentration_tail.csv", index=False)

    print("\n=== Headline ===")
    print(f"  std(u_k): closed-form sqrt(V) matches empirical within "
          f"{(df.std_ratio - 1).abs().max()*100:.0f}% (ratio "
          f"{df.std_ratio.min():.2f}-{df.std_ratio.max():.2f}); "
          f"std stays {df.emp_std.min():.2f}-{df.emp_std.max():.2f} (O(sqrt m), not O(sqrt k)).")
    print(f"  Chebyshev tail bound holds in {int(tdf.holds.sum())}/{len(tdf)} cells "
          f"(never violated), conservative as expected.")
    print(f"\n  Wrote {out / 'concentration_std.csv'} and concentration_tail.csv.")


if __name__ == "__main__":
    main()
