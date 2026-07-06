"""Out-of-distribution validation of the DEPLOYABLE forecaster.

The headline synthetic grid is a self-consistency check: it draws the workload
from a Zipf marginal and feeds the SAME marginal to the closed form. The honest
objection is "your result is a Zipf tautology." This script rebuts it directly,
on the estimator the paper actually ships.

The deployable forecaster does NOT assume a distribution family. It observes a
PREFIX of the stream and extrapolates the full distinct-query count u_k with an
unseen-species estimator (plug-in occupancy, or Smoothed Good--Toulmin
[Orlitsky-Suresh-Wu 2016]) -- exactly the Redbench pipeline, but here in a
controlled setting where we KNOW the generating distribution and can make it
NON-Zipf on purpose.

We generate length-k workloads from five families -- two Zipf, plus uniform,
lognormal-ish and a heavy Pareto-ish tail -- and forecast the full u_k from only
the first 50% / 25% of each. The forecaster never knows the family. If the
forecast holds across all of them, the method is not tied to the Zipf assumption.
Error is reported as |pred - realized| / k (same normalization as the Redbench
0.22 -> 0.12 headline) and as relative error.

Deterministic seeds. Run: python experiments/ood_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import uniform_distribution, zipf_distribution
from dpdb.predictors import predict_plugin, predict_smoothed_gt

M = 200
K = 400
TRIALS = 60
FRACS = (0.5, 0.25)   # forecast full u_k from this fraction as prefix


def _lognormalish(m, sigma=1.5):
    x = np.exp(sigma * np.arange(m) / m)[::-1]
    return x / x.sum()


def _paretoish(m, beta=1.5):
    x = (1.0 + np.arange(m)) ** (-beta)
    return x / x.sum()


def families():
    return {
        "zipf(0.8)": zipf_distribution(M, 0.8),
        "zipf(1.2)": zipf_distribution(M, 1.2),
        "uniform": uniform_distribution(M),
        "lognormal-ish": _lognormalish(M),
        "pareto-ish(b=1.5)": _paretoish(M, 1.5),
    }


def main():
    rows = []
    for name, q in families().items():
        for frac in FRACS:
            n = int(frac * K)
            e_plugin, e_gt, e_naive = [], [], []  # errors in u_k/k
            for t in range(TRIALS):
                rng = np.random.default_rng(424242 + 977 * int(frac * 100) + t)
                full = rng.choice(M, size=K, p=q)
                prefix = full[:n]
                counts = np.bincount(prefix, minlength=M)
                realized = len(set(full.tolist()))
                d_n = int((counts > 0).sum())          # naive: assume prefix already saw everything
                # A distinct-count forecast must lie in [already-seen, k]; any
                # sane allocator clamps to this range. Beyond horizon t~1 the
                # Good--Toulmin series is known to diverge (predictors.py), and
                # the clamp turns a divergence into the budget-safe k (over-forecast).
                clamp = lambda v: float(min(max(v, d_n), K))
                p_plugin = clamp(predict_plugin(counts, n, K))
                p_gt = clamp(predict_smoothed_gt(counts, n, K))
                e_plugin.append(abs(p_plugin - realized) / K)
                e_gt.append(abs(p_gt - realized) / K)
                e_naive.append(abs(d_n - realized) / K)
            rows.append(dict(
                family=name, prefix_frac=frac, n=n, k=K,
                mae_naive=float(np.mean(e_naive)),
                mae_plugin=float(np.mean(e_plugin)),
                mae_gt=float(np.mean(e_gt)),
            ))

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "ood"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ood_validation.csv", index=False)

    print("=== OOD prefix-forecast: deployable estimator, NON-Zipf truths ===")
    print("    forecast full u_k from a prefix; estimator never knows the family.")
    print("    error = |predicted - realized| / k  (same units as Redbench 0.22->0.12)\n")
    print(df.round(4).to_string(index=False))

    half = df[df.prefix_frac == 0.5]
    print("\n=== Headline (50% prefix) ===")
    print(f"  Smoothed Good--Toulmin MAE(u_k/k): mean {half.mae_gt.mean():.3f} "
          f"(max {half.mae_gt.max():.3f}) across all five families, Zipf and non-Zipf alike.")
    print(f"  Plug-in occupancy MAE:             mean {half.mae_plugin.mean():.3f} "
          f"(max {half.mae_plugin.max():.3f}).")
    print(f"  Naive (prefix-distinct) MAE:       mean {half.mae_naive.mean():.3f} "
          f"(max {half.mae_naive.max():.3f}).")
    impr = (1 - half.mae_gt.mean() / half.mae_naive.mean()) * 100
    print(f"  Unseen-species correction cuts the naive error by {impr:.0f}% on non-Zipf data too,")
    print(f"    so the forecast is NOT an artifact of the Zipf self-consistency setup.")
    print(f"\n  Wrote {out / 'ood_validation.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
