"""Empirical validation of the R2 analytical model.

For each (workload distribution, k) cell, we:
  1. Compute analytical prediction E[u_k] from model.py
  2. Run N trials of the workload through the real middleware
  3. Compare empirical mean(u_k) ± 95% CI to the analytical prediction

The output is the model-vs-empirical plot the revision brief requires.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows
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
from dpdb.model import (
    expected_unique_queries,
    budget_savings_ratio,
    zipf_distribution,
)
from dpdb.workload_gen import (
    ADULT_AGE_BAND_TEMPLATES,
    generate_zipf_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_single_trial(
    config: Config,
    queries: list[str],
    eps_per_query: float,
    seed: int,
) -> dict:
    """Run a workload through the workload-aware middleware once and return metrics."""
    np.random.seed(seed)
    mw = DPMiddleware(config, mode=ExecutionMode.WORKLOAD_DP)
    cache_hits = 0
    for sql in queries:
        try:
            r = mw.execute(sql, epsilon=eps_per_query)
            if r.cache_hit:
                cache_hits += 1
        except Exception:
            pass
    s = mw.budget_summary()
    # Empirical unique (template, param) pairs = number of cache misses
    # = consumed_epsilon / eps_per_query (each unique query costs exactly eps_per_query)
    empirical_unique = s["consumed_epsilon"] / eps_per_query if eps_per_query > 0 else 0
    return {
        "consumed_epsilon": s["consumed_epsilon"],
        "cache_hits": cache_hits,
        "unique_queries": empirical_unique,
        "total_queries": len(queries),
    }


def validate_model(
    config: Config,
    output_dir: Path,
    eps_per_query: float = 1.0,
    n_trials: int = 30,
):
    """Run model-vs-empirical validation across (alpha, k) grid."""
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    k_values = [10, 25, 50, 100]
    m = len(ADULT_AGE_BAND_TEMPLATES)

    records = []
    print(f"Running validation: {len(alphas)} alphas x {len(k_values)} k-values x {n_trials} trials")
    print(f"Template pool size m = {m}, eps_q = {eps_per_query}")
    for alpha in alphas:
        p = zipf_distribution(m, alpha)
        predicted_eu_at_k = {k: expected_unique_queries(p, k) for k in k_values}
        predicted_savings = {k: budget_savings_ratio(p, k) for k in k_values}
        for k in k_values:
            for trial in range(n_trials):
                seed = trial * 17 + int(alpha * 1000) + k
                queries, _ = generate_zipf_workload(
                    ADULT_AGE_BAND_TEMPLATES, alpha=alpha, k=k, seed=seed,
                )
                m_res = run_single_trial(config, queries, eps_per_query, seed=seed)
                records.append({
                    "alpha": alpha,
                    "k": k,
                    "trial": trial,
                    "empirical_eps": m_res["consumed_epsilon"],
                    "empirical_unique": m_res["unique_queries"],
                    "empirical_cache_hits": m_res["cache_hits"],
                    "empirical_savings": 1.0 - m_res["consumed_epsilon"] / (k * eps_per_query),
                    "predicted_unique": predicted_eu_at_k[k],
                    "predicted_savings": predicted_savings[k],
                    "predicted_eps": predicted_eu_at_k[k] * eps_per_query,
                })
            print(f"  alpha={alpha}, k={k}: model E[u_k]={predicted_eu_at_k[k]:.2f}")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_validation.csv", index=False)
    print(f"\nSaved: {output_dir / 'model_validation.csv'}")
    return df


def plot_validation(df: pd.DataFrame, output_dir: Path):
    """Generate the model-vs-empirical plots required by the revision brief."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: E[u_k] vs empirical mean ± 95% CI, faceted by alpha
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    alphas = sorted(df["alpha"].unique())
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        sub = df[df["alpha"] == alpha]
        grp = sub.groupby("k").agg(
            empirical_mean=("empirical_unique", "mean"),
            empirical_std=("empirical_unique", "std"),
            predicted=("predicted_unique", "first"),
            n=("empirical_unique", "size"),
        ).reset_index()
        ci = 1.96 * grp["empirical_std"] / np.sqrt(grp["n"])
        ax.errorbar(grp["k"], grp["empirical_mean"], yerr=ci, fmt="o",
                    label="Empirical (mean ± 95% CI)", color="C0", markersize=8)
        ax.plot(grp["k"], grp["predicted"], "--", label="Model E[u_k]",
                color="C3", linewidth=2)
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("k (workload length)")
        if i % 3 == 0:
            ax.set_ylabel("Unique queries u_k")
        ax.legend(loc="best", fontsize=9)
    plt.suptitle("R2 Model Validation: Predicted vs Empirical E[u_k] (Adult age-band workload)")
    plt.tight_layout()
    plt.savefig(output_dir / "model_validation_unique.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "model_validation_unique.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Budget savings ratio vs alpha at fixed k
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k]
        grp = sub.groupby("alpha").agg(
            emp_mean=("empirical_savings", "mean"),
            emp_std=("empirical_savings", "std"),
            pred=("predicted_savings", "first"),
            n=("empirical_savings", "size"),
        ).reset_index()
        ci = 1.96 * grp["emp_std"] / np.sqrt(grp["n"])
        line, = ax.plot(grp["alpha"], grp["pred"], "--", label=f"Model k={k}")
        ax.errorbar(grp["alpha"], grp["emp_mean"], yerr=ci, fmt="o",
                    color=line.get_color(), label=f"Empirical k={k}")
    ax.set_xlabel(r"Zipf skew parameter $\alpha$")
    ax.set_ylabel("Budget savings ratio S(k)")
    ax.set_title("Budget Savings: Model Prediction vs Empirical (Adult age-band workload)")
    ax.legend(loc="best", ncol=2, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "model_validation_savings.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "model_validation_savings.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Limit verification — alpha=inf and alpha=0 special cases
    fig, ax = plt.subplots(figsize=(10, 6))
    k_grid = np.arange(2, 200)
    # Limit A: 1 - 1/k
    ax.plot(k_grid, 1 - 1 / k_grid, "k-", label=r"Limit A: $1 - 1/k$ (perfect repetition)")
    # Limit B: 0
    ax.axhline(0, color="gray", linestyle=":", label="Limit B: 0 (uniform, m→∞)")
    # Empirical from data
    for alpha in [0.0, 1.0, 3.0]:
        sub = df[df["alpha"] == alpha]
        grp = sub.groupby("k")["empirical_savings"].mean()
        ax.plot(grp.index, grp.values, "o-", label=f"Empirical alpha={alpha}")
    ax.set_xlabel("k")
    ax.set_ylabel("S(k)")
    ax.set_title("Empirical Savings Bracketed by Model Limits")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_validation_limits.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "model_validation_limits.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3 figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/model_validation")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--load-existing", action="store_true",
                        help="Skip benchmark, just regenerate plots from existing CSV")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = Path(args.output)

    if args.load_existing and (output_dir / "model_validation.csv").exists():
        df = pd.read_csv(output_dir / "model_validation.csv")
        print(f"Loaded existing results: {len(df)} rows")
    else:
        df = validate_model(config, output_dir, args.epsilon, args.trials)

    plot_validation(df, output_dir)


if __name__ == "__main__":
    main()
