"""Aggregate all experimental results into one summary CSV + markdown report.

Loads every experiment's CSV, computes the headline metrics, and writes:
  - results/ALL_RESULTS.csv  : long-form table of every metric
  - results/REPORT.md        : human-readable summary for paper inclusion
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd


def safe_read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def main():
    root = Path("results")
    summaries = {}

    # Original W1-W4 benchmark
    bench = safe_read(root / "results.csv")
    if bench is not None:
        summaries["original_benchmark"] = bench

    # Model validation (alpha x k, 30 trials)
    mv = safe_read(root / "model_validation" / "model_validation.csv")
    if mv is not None:
        mv["err_abs"] = (mv["predicted_unique"] - mv["empirical_unique"]).abs()
        mv["err_pct"] = 100 * mv["err_abs"] / mv["predicted_unique"].clip(lower=1e-9)
        summaries["model_validation"] = mv

    # Leakage
    mia = safe_read(root / "leakage" / "mia_results.csv")
    rec = safe_read(root / "leakage" / "reconstruction_results.csv")
    if mia is not None:
        summaries["mia"] = mia
    if rec is not None:
        summaries["reconstruction"] = rec

    # Temporal validation
    tv = safe_read(root / "temporal" / "temporal_validation.csv")
    if tv is not None:
        summaries["temporal"] = tv

    # Semantic L2
    sem = safe_read(root / "semantic" / "semantic_validation.csv")
    if sem is not None:
        summaries["semantic"] = sem

    # Extended sweeps
    ext_a = safe_read(root / "extended" / "extended_alpha.csv")
    ext_e = safe_read(root / "extended" / "epsilon_sweep.csv")
    ext_k = safe_read(root / "extended" / "large_k.csv")
    if ext_a is not None: summaries["extended_alpha"] = ext_a
    if ext_e is not None: summaries["epsilon_sweep"] = ext_e
    if ext_k is not None: summaries["large_k"] = ext_k

    # Full campaign
    fc = safe_read(root / "full_campaign" / "full_campaign_SF1.csv")
    if fc is not None: summaries["full_campaign"] = fc

    # SF=10 cross-validation
    sf10 = safe_read(root / "sf10" / "sf10_validation.csv")
    if sf10 is not None: summaries["sf10_validation"] = sf10

    # ==================================================================
    # Write the report
    # ==================================================================
    report = [
        "# Comprehensive Experimental Report",
        "",
        f"Generated from {len(summaries)} experiment CSVs.",
        "",
    ]

    if "model_validation" in summaries:
        report.append("## Model Validation (R6 main): alpha sweep at fixed k")
        report.append("")
        df = summaries["model_validation"]
        report.append(f"Total trials: {len(df)} ({df['alpha'].nunique()} alphas x {df['k'].nunique()} k-values x {df['trial'].max()+1} trials)")
        report.append("")
        report.append("Mean predicted vs empirical mean by (alpha, k) -- the model claim is mean-vs-mean accuracy:")
        report.append("")
        report.append("```")
        g = df.groupby(["alpha", "k"]).agg(
            pred=("predicted_unique", "first"),
            emp_mean=("empirical_unique", "mean"),
            emp_std=("empirical_unique", "std"),
        ).round(3)
        g["mean_err_abs"] = (g["pred"] - g["emp_mean"]).abs().round(3)
        g["mean_err_pct"] = (100 * g["mean_err_abs"] / g["pred"].clip(lower=1e-9)).round(2)
        report.append(g.to_string())
        report.append("```")
        worst_err = g["mean_err_pct"].max()
        ge_3 = (g["mean_err_pct"] >= 3.0).sum()
        report.append(f"\nCells with |pred - empirical_mean| / pred >= 3%: {ge_3} / {len(g)}")
        report.append(f"Worst case mean-vs-mean relative error: {worst_err:.2f}%")
        report.append("")

    if "mia" in summaries:
        report.append("## Membership Inference Attack (R4)")
        report.append("")
        report.append("```")
        report.append(summaries["mia"].round(4).to_string(index=False))
        report.append("```")
        report.append("")

    if "reconstruction" in summaries:
        report.append("## Reconstruction Attack (R4)")
        report.append("")
        report.append("```")
        df = summaries["reconstruction"][["eps_per_query", "mean_abs_error",
                                          "p95_abs_error", "max_abs_error"]]
        report.append(df.round(3).to_string(index=False))
        report.append("```")
        report.append("")

    if "temporal" in summaries:
        report.append("## Temporal Validation (R3)")
        report.append("")
        report.append("```")
        df = summaries["temporal"]
        g = df.groupby(["experiment", "tau", "lambda"]).agg(
            emp=("empirical_eps", "mean"),
            pred=("predicted_eps", "first"),
            expired=("expired", "mean"),
            invalidated=("update_evicted", "mean"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "extended_alpha" in summaries:
        report.append("## Extended Alpha Sweep (alpha up to 10, k=200)")
        report.append("")
        report.append("```")
        df = summaries["extended_alpha"]
        g = df.groupby("alpha").agg(
            pred=("predicted_unique", "first"),
            emp=("empirical_unique", "mean"),
            std=("empirical_unique", "std"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "epsilon_sweep" in summaries:
        report.append("## Epsilon Sweep Validation (eps in {0.1, 0.5, 1.0, 2.0})")
        report.append("")
        report.append("```")
        df = summaries["epsilon_sweep"]
        df["err_pct"] = 100 * (df["predicted_unique"] - df["empirical_unique"]).abs() / df["predicted_unique"]
        g = df.groupby(["eps_q", "k"]).agg(
            pred=("predicted_unique", "first"),
            emp=("empirical_unique", "mean"),
            err_pct=("err_pct", "mean"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "large_k" in summaries:
        report.append("## Large-k Sweep (k up to 500)")
        report.append("")
        report.append("```")
        df = summaries["large_k"]
        g = df.groupby("k").agg(
            pred=("predicted_unique", "first"),
            emp=("empirical_unique", "mean"),
            std=("empirical_unique", "std"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "semantic" in summaries:
        report.append("## Semantic L2 Cache Validation")
        report.append("")
        report.append("```")
        df = summaries["semantic"]
        g = df.groupby(["alpha", "mode"]).agg(
            eps=("consumed_epsilon", "mean"),
            exact_hits=("exact_hits", "mean"),
            sem_hits=("semantic_hits", "mean"),
            err=("mean_abs_error", "mean"),
            lat_ms=("avg_latency_ms", "mean"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "full_campaign" in summaries:
        report.append("## Full Benchmark Campaign (6 workloads x 4 eps x 3 modes x 30 trials)")
        report.append("")
        df = summaries["full_campaign"]
        report.append(f"Total cells: {len(df)} trials")
        report.append("")
        report.append("Budget consumption by (workload, mode) at eps_q=1.0:")
        report.append("")
        report.append("```")
        g = df[df["eps_per_query"] == 1.0].groupby(["workload", "mode"]).agg(
            budget=("consumed_eps", "mean"),
            cache=("cache_hits", "mean"),
            err=("mean_abs_error", "mean"),
            n_ans=("n_answered", "mean"),
        ).round(2)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    if "sf10_validation" in summaries:
        report.append("## Cross-Scale Validation (SF=1 vs SF=10)")
        report.append("")
        report.append("```")
        df = summaries["sf10_validation"]
        g = df.groupby(["scale_factor", "template_set", "alpha"]).agg(
            pred=("predicted_unique", "first"),
            emp=("empirical_unique", "mean"),
        ).round(3)
        report.append(g.to_string())
        report.append("```")
        report.append("")

    Path("results/REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote results/REPORT.md ({len(report)} lines)")

    # Also dump a long-form table of every loaded df
    all_dfs = []
    for name, df in summaries.items():
        d = df.copy()
        d["source"] = name
        all_dfs.append(d)
    if all_dfs:
        combined = pd.concat(all_dfs, axis=0, ignore_index=True, sort=False)
        combined.to_csv("results/ALL_RESULTS.csv", index=False)
        print(f"Wrote results/ALL_RESULTS.csv ({len(combined)} rows)")


if __name__ == "__main__":
    main()
