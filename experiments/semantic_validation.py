"""Semantic L2 cache validation experiment.

Compares three strategies on parametric workloads where exact-repeat caching
helps only partially:

  - NAIVE_DP:     no caching, every query consumes eps_q
  - WORKLOAD_DP:  L1 exact-match cache (template hash + param hash)
  - SEMANTIC_DP:  L1 + L2 semantic cache (Tree Kernel + AST Embedding)

The semantic L2 layer catches structurally equivalent queries that differ only
in literal values inside the WHERE clause -- a scenario that exact-match cache
misses but that is privacy-safe to reuse via the post-processing property
when the queries are alpha-equivalent.

Measures:
  - Budget consumed
  - Cache hit rate (exact vs semantic)
  - Per-query latency (with embedding overhead)
  - Approximation error introduced by semantic reuse (which is NOT
    privacy-free when queries differ in their true answer)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.workload_gen import (
    ADULT_AGE_BAND_TEMPLATES,
    generate_zipf_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_trial(
    config: Config,
    mode: ExecutionMode,
    queries: list[str],
    eps_per_query: float,
    seed: int,
) -> dict:
    np.random.seed(seed)
    t0 = time.perf_counter()
    mw = DPMiddleware(config, mode=mode)

    # Get ground truth for utility comparison
    mw_exact = DPMiddleware(config, mode=ExecutionMode.EXACT, db=mw.db)
    true_values = []
    noisy_values = []
    cache_hits = 0
    semantic_hits = 0

    for sql in queries:
        try:
            t_q = mw_exact.execute(sql)
            true_v = float(t_q.rows[0][0]) if t_q.rows else 0.0
        except Exception:
            true_v = 0.0

        try:
            r = mw.execute(sql, epsilon=eps_per_query)
            if r.rows and r.rows[0]:
                noisy_v = float(r.rows[0][0])
            else:
                noisy_v = 0.0
            if r.cache_hit:
                cache_hits += 1
        except Exception:
            noisy_v = 0.0

        true_values.append(true_v)
        noisy_values.append(noisy_v)

    s = mw.budget_summary()
    semantic_hits = s.get("semantic_hits", 0)
    wall_time = time.perf_counter() - t0

    abs_errors = [abs(n - t) for n, t in zip(noisy_values, true_values)]
    return {
        "mode": mode.value,
        "consumed_epsilon": s["consumed_epsilon"],
        "cache_hits": cache_hits,
        "exact_hits": cache_hits - semantic_hits,
        "semantic_hits": semantic_hits,
        "total_queries": len(queries),
        "mean_abs_error": np.mean(abs_errors),
        "max_abs_error": np.max(abs_errors),
        "wall_time_sec": wall_time,
        "avg_latency_ms": (wall_time / len(queries)) * 1000,
    }


def run_semantic_validation(
    config: Config,
    output_dir: Path,
    eps_per_query: float = 1.0,
    n_trials: int = 10,
):
    """For each Zipf alpha, run three modes and compare."""
    alphas = [0.0, 0.5, 1.0, 2.0]
    k = 50
    modes = [
        ExecutionMode.NAIVE_DP,
        ExecutionMode.WORKLOAD_DP,
        ExecutionMode.SEMANTIC_DP,
    ]
    templates = ADULT_AGE_BAND_TEMPLATES

    records = []
    print(f"Running semantic L2 validation: alphas={alphas}, k={k}, trials={n_trials}")
    for alpha in alphas:
        for trial in range(n_trials):
            seed = trial * 31 + int(alpha * 1000)
            queries, _ = generate_zipf_workload(templates, alpha=alpha, k=k, seed=seed)
            for mode in modes:
                r = run_trial(config, mode, queries, eps_per_query, seed)
                r["alpha"] = alpha
                r["trial"] = trial
                r["k"] = k
                records.append(r)
            print(f"  alpha={alpha}, trial={trial+1}/{n_trials} done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "semantic_validation.csv", index=False)
    return df


def plot_semantic_validation(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Budget consumption across modes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db", "semantic_dp": "#2ecc71"}
    grp = df.groupby(["alpha", "mode"])["consumed_epsilon"].mean().reset_index()
    sns.barplot(data=grp, x="alpha", y="consumed_epsilon", hue="mode",
                palette=palette, ax=axes[0])
    axes[0].set_title("Privacy Budget Consumption by Mode")
    axes[0].set_xlabel(r"Zipf $\alpha$")
    axes[0].set_ylabel(r"$\varepsilon$ consumed (lower = better)")

    grp2 = df.groupby(["alpha", "mode"])["mean_abs_error"].mean().reset_index()
    sns.barplot(data=grp2, x="alpha", y="mean_abs_error", hue="mode",
                palette=palette, ax=axes[1])
    axes[1].set_title("Mean Absolute Error by Mode")
    axes[1].set_xlabel(r"Zipf $\alpha$")
    axes[1].set_ylabel("Mean abs error per query")

    plt.tight_layout()
    plt.savefig(output_dir / "semantic_validation_main.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "semantic_validation_main.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Cache hit decomposition (exact vs semantic)
    fig, ax = plt.subplots(figsize=(10, 6))
    wa_data = df[df["mode"] == "workload_dp"].groupby("alpha")["exact_hits"].mean()
    sem_exact = df[df["mode"] == "semantic_dp"].groupby("alpha")["exact_hits"].mean()
    sem_l2 = df[df["mode"] == "semantic_dp"].groupby("alpha")["semantic_hits"].mean()

    alphas = sorted(df["alpha"].unique())
    x = np.arange(len(alphas))
    width = 0.35
    ax.bar(x - width/2, wa_data.values, width,
           label="Workload-aware L1 only (exact hits)",
           color="#3498db")
    ax.bar(x + width/2, sem_exact.values, width,
           label="Semantic mode: L1 exact hits", color="#3498db", alpha=0.7)
    ax.bar(x + width/2, sem_l2.values, width, bottom=sem_exact.values,
           label="Semantic mode: L2 semantic hits", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"Zipf $\alpha$")
    ax.set_ylabel("Cache hits per workload (k=50)")
    ax.set_title("Cache hit decomposition: exact vs semantic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_cache_hits.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "semantic_cache_hits.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/semantic")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    df = run_semantic_validation(config, Path(args.output), args.epsilon, args.trials)

    print("\n=== Summary ===")
    print(df.groupby(["alpha", "mode"]).agg(
        eps=("consumed_epsilon", "mean"),
        exact=("exact_hits", "mean"),
        semantic=("semantic_hits", "mean"),
        err=("mean_abs_error", "mean"),
        lat_ms=("avg_latency_ms", "mean"),
    ).round(3).to_string())

    plot_semantic_validation(df, Path(args.output))


if __name__ == "__main__":
    main()
