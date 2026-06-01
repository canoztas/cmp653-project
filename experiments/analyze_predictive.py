"""Committed analysis of the predictive-allocator comparison (paper Table
tab:predictive). Loads results/predictive/predictive_comparison.csv, computes the
per-alpha mean MAE, standard error, paired t-test (on per-trial means, the
correct independent unit) and win-count of PREDICTIVE vs NAIVE, and writes them to
results/predictive/predictive_stats.csv so the paper's numbers are reproducible.

Run: python experiments/analyze_predictive.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).parent.parent / "results" / "predictive" / "predictive_comparison.csv"


def main():
    df = pd.read_csv(CSV)
    n_trials = df["trial"].nunique()
    rows = []
    for alpha, g in df[df["mode"].isin(["naive_dp", "predictive_dp"])].groupby("alpha"):
        nv = g[g["mode"] == "naive_dp"].sort_values("trial")["mean_abs_error"].values
        pr = g[g["mode"] == "predictive_dp"].sort_values("trial")["mean_abs_error"].values
        m = min(len(nv), len(pr))
        nv, pr = nv[:m], pr[:m]
        t, p = stats.ttest_rel(nv, pr)
        ans = g[g["mode"] == "predictive_dp"]["n_answered"].mean()
        rows.append({
            "alpha": alpha, "n_trials": m,
            "naive_mae": round(float(nv.mean()), 2),
            "pred_mae": round(float(pr.mean()), 2),
            "pred_se": round(float(pr.std(ddof=1) / np.sqrt(m)), 2),
            "reduction_pct": round(100 * (nv.mean() - pr.mean()) / nv.mean(), 1),
            "paired_t": round(float(t), 2), "p_value": round(float(p), 3),
            "pred_wins": int((pr < nv).sum()),
            "answered": round(float(ans), 0),
        })
    out = pd.DataFrame(rows)
    out.to_csv(CSV.parent / "predictive_stats.csv", index=False)
    print(f"n_trials = {n_trials}")
    print(out.to_string(index=False))
    sig = out[out.p_value < 0.05]
    print(f"\nSignificant (p<0.05) only at alpha in {list(sig.alpha)} "
          f"-> report the gain as a low-skew, directional result.")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
