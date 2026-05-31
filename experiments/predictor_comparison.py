"""Benchmark: which estimator best predicts u_k from a short warmup prefix?

The predictive allocator sees a prefix of ``n`` queries and must predict the
number of distinct templates ``u_k`` over the full workload of ``k`` queries.
We compare the current plug-in occupancy estimator against the unseen-species
estimators (Good-Toulmin, Smoothed Good-Toulmin) and the Chao1 richness
reference, across Zipf skew and prefix length. Deterministic seeds.

Run: python experiments/predictor_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import zipf_distribution
from dpdb.predictors import PREDICTORS

M = 50            # number of distinct templates available (large -> u_k < M, real extrapolation)
K = 100           # full workload length
N_GRID = [20, 33, 50, 66]    # warmup prefix lengths (horizon factor t=(K-n)/n: 4, 2, 1, 0.5)
ALPHA_GRID = [0.5, 1.0, 1.5, 2.0]   # skewed workloads (u_k well below M)
TRIALS = 40
SEED = 7


def run():
    rng = np.random.default_rng(SEED)
    rows = []
    for alpha in ALPHA_GRID:
        p = zipf_distribution(M, alpha)
        for n in N_GRID:
            acc = {name: [] for name in PREDICTORS}
            true_list = []
            for _ in range(TRIALS):
                draws = rng.choice(M, size=K, p=p)
                u_k = len(np.unique(draws))           # ground truth
                prefix = draws[:n]
                counts = np.bincount(prefix, minlength=M)
                counts = counts[counts > 0]
                true_list.append(u_k)
                d_n = int((counts > 0).sum())
                for name, fn in PREDICTORS.items():
                    pred = fn(counts, n, K)
                    # allocator guardrail: a real Û is clamped to [distinct seen, k]
                    if not np.isfinite(pred):
                        pred = float(K)
                    pred = min(max(pred, d_n), K)
                    acc[name].append(pred)
            mean_true = float(np.mean(true_list))
            for name in PREDICTORS:
                preds = np.array(acc[name], dtype=float)
                # clip pathological divergence for reporting (raw GT can explode)
                rel = (preds - np.array(true_list)) / np.array(true_list)
                rows.append({
                    "alpha": alpha, "n": n, "t": round((K - n) / n, 1),
                    "predictor": name,
                    "mean_pred": round(float(np.mean(preds)), 2),
                    "mean_true": round(mean_true, 2),
                    "signed_rel_err_pct": round(float(np.mean(rel)) * 100, 1),
                    "abs_rel_err_pct": round(float(np.mean(np.abs(rel))) * 100, 1),
                })
    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "predictors"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictor_comparison.csv", index=False)

    # ---- summary: mean absolute relative error, predictor x n (avg over alpha)
    print("\n=== Mean ABS relative error of u_k prediction (%), averaged over alpha ===")
    piv = df.pivot_table(index="predictor", columns="n",
                         values="abs_rel_err_pct", aggfunc="mean").round(1)
    piv = piv.reindex(["plugin", "good_toulmin", "smoothed_gt", "chao1"])
    print(piv.to_string())

    print("\n=== Signed bias (%), averaged over alpha (negative = under-predicts) ===")
    pivb = df.pivot_table(index="predictor", columns="n",
                          values="signed_rel_err_pct", aggfunc="mean").round(1)
    pivb = pivb.reindex(["plugin", "good_toulmin", "smoothed_gt", "chao1"])
    print(pivb.to_string())

    # winner per (alpha, n) among the horizon-aware predictors (exclude chao1)
    cand = df[df.predictor != "chao1"]
    best = (cand.loc[cand.groupby(["alpha", "n"]).abs_rel_err_pct.idxmin()]
            .predictor.value_counts())
    print("\n=== Best horizon-aware predictor, # of (alpha,n) cells won ===")
    print(best.to_string())
    print(f"\nSaved {out / 'predictor_comparison.csv'}")
    return df


if __name__ == "__main__":
    run()
