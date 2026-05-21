"""Extended model validation sweeps for the final paper.

Three sweeps:
  - Epsilon sweep: model validation across eps in {0.1, 0.5, 1.0, 2.0}
  - Wider alpha range: alpha in {0, 0.5, 1, 1.5, 2, 3, 5, 10}, larger k
  - Larger workloads: k in {25, 50, 100, 250, 500}

Outputs full grid + 2 summary figures.
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
from dpdb.model import expected_unique_queries, zipf_distribution
from dpdb.workload_gen import ADULT_AGE_BAND_TEMPLATES, generate_zipf_workload

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_trial(config, queries, eps_q, seed):
    np.random.seed(seed)
    import copy
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = 1000.0  # remove budget cap for measurement
    mw = DPMiddleware(cfg, mode=ExecutionMode.WORKLOAD_DP)
    for sql in queries:
        try:
            mw.execute(sql, epsilon=eps_q)
        except Exception:
            pass
    s = mw.budget_summary()
    return s["consumed_epsilon"] / eps_q  # = empirical unique count


def run_extended_alpha_sweep(config, output_dir, n_trials=30):
    """Wider alpha sweep at k=200, more alpha values."""
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    k = 200
    eps_q = 1.0
    m = len(ADULT_AGE_BAND_TEMPLATES)

    records = []
    print(f"[Extended alpha sweep] {len(alphas)} alphas x {n_trials} trials at k={k}")
    for alpha in alphas:
        p = zipf_distribution(m, alpha)
        pred = expected_unique_queries(p, k)
        for trial in range(n_trials):
            seed = trial * 41 + int(alpha * 100)
            queries, _ = generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES,
                                                alpha=alpha, k=k, seed=seed)
            emp = run_trial(config, queries, eps_q, seed)
            records.append({
                "alpha": alpha, "k": k, "trial": trial,
                "empirical_unique": emp, "predicted_unique": pred,
            })
        print(f"  alpha={alpha}: pred={pred:.2f}")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "extended_alpha.csv", index=False)
    return df


def run_epsilon_sweep_validation(config, output_dir, n_trials=30):
    """Eps sweep at fixed alpha for model validation."""
    eps_values = [0.1, 0.5, 1.0, 2.0]
    k_values = [25, 50, 100, 250]
    alpha = 1.0
    p = zipf_distribution(len(ADULT_AGE_BAND_TEMPLATES), alpha)

    records = []
    print(f"[Epsilon sweep] eps in {eps_values} x k in {k_values} x {n_trials} trials")
    for eps_q in eps_values:
        for k in k_values:
            pred = expected_unique_queries(p, k)
            for trial in range(n_trials):
                seed = trial * 53 + int(eps_q * 100) + k
                queries, _ = generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES,
                                                    alpha=alpha, k=k, seed=seed)
                emp = run_trial(config, queries, eps_q, seed)
                records.append({
                    "eps_q": eps_q, "k": k, "trial": trial, "alpha": alpha,
                    "empirical_unique": emp, "predicted_unique": pred,
                })
            print(f"  eps={eps_q}, k={k}: pred={pred:.2f}")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "epsilon_sweep.csv", index=False)
    return df


def run_large_k_sweep(config, output_dir, n_trials=15):
    """Large-k sweep at fixed alpha = 1.0."""
    k_values = [25, 50, 100, 250, 500]
    alpha = 1.0
    eps_q = 1.0
    p = zipf_distribution(len(ADULT_AGE_BAND_TEMPLATES), alpha)

    records = []
    print(f"[Large k sweep] k in {k_values} x {n_trials} trials")
    for k in k_values:
        pred = expected_unique_queries(p, k)
        for trial in range(n_trials):
            seed = trial * 67 + k
            queries, _ = generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES,
                                                alpha=alpha, k=k, seed=seed)
            emp = run_trial(config, queries, eps_q, seed)
            records.append({
                "k": k, "trial": trial, "alpha": alpha,
                "empirical_unique": emp, "predicted_unique": pred,
            })
        print(f"  k={k}: pred={pred:.2f}")
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "large_k.csv", index=False)
    return df


def plot_extended(alpha_df, eps_df, k_df, output_dir):
    """Two summary figures."""
    # Figure: alpha sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    g = alpha_df.groupby("alpha").agg(
        emp_mean=("empirical_unique", "mean"),
        emp_std=("empirical_unique", "std"),
        pred=("predicted_unique", "first"),
        n=("empirical_unique", "size"),
    ).reset_index()
    ci = 1.96 * g["emp_std"] / np.sqrt(g["n"])
    ax.errorbar(g["alpha"], g["emp_mean"], yerr=ci, fmt="o",
                markersize=10, label="Empirical (mean +/- 95% CI)", color="C0")
    ax.plot(g["alpha"], g["pred"], "--", linewidth=2,
            label="Model E[u_k]", color="C3")
    ax.set_xlabel(r"Zipf $\alpha$ (skew parameter)")
    ax.set_ylabel(r"Unique queries $u_k$ at $k=200$")
    ax.set_title("Extended alpha sweep — model vs empirical")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "extended_alpha_sweep.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "extended_alpha_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure: epsilon sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    eps_levels = sorted(eps_df["eps_q"].unique())
    for eps_q in eps_levels:
        sub = eps_df[eps_df["eps_q"] == eps_q]
        g = sub.groupby("k").agg(
            emp_mean=("empirical_unique", "mean"),
            emp_std=("empirical_unique", "std"),
            pred=("predicted_unique", "first"),
            n=("empirical_unique", "size"),
        ).reset_index()
        ci = 1.96 * g["emp_std"] / np.sqrt(g["n"])
        line = ax.errorbar(g["k"], g["emp_mean"], yerr=ci, fmt="o",
                           markersize=8, label=f"Empirical eps={eps_q}")
    # Model is the same for any eps (eps cancels out in this metric)
    g0 = eps_df.groupby("k")["predicted_unique"].first()
    ax.plot(g0.index, g0.values, "k--", linewidth=2, label="Model E[u_k] (eps-independent)")
    ax.set_xlabel("Workload length k")
    ax.set_ylabel("Unique queries u_k")
    ax.set_title("Epsilon sweep — model is independent of eps_q in this metric")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "extended_eps_sweep.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "extended_eps_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure: large k sweep with concentration
    fig, ax = plt.subplots(figsize=(10, 6))
    g = k_df.groupby("k").agg(
        emp_mean=("empirical_unique", "mean"),
        emp_std=("empirical_unique", "std"),
        pred=("predicted_unique", "first"),
        n=("empirical_unique", "size"),
    ).reset_index()
    ci = 1.96 * g["emp_std"] / np.sqrt(g["n"])
    ax.errorbar(g["k"], g["emp_mean"], yerr=ci, fmt="o",
                markersize=8, label="Empirical", color="C0")
    ax.plot(g["k"], g["pred"], "--", linewidth=2, label="Model E[u_k]", color="C3")
    # Show m horizontal line (max possible u_k)
    m = 7
    ax.axhline(m, color="gray", linestyle=":", label=f"Saturation at m={m}")
    ax.set_xlabel("Workload length k (log)")
    ax.set_ylabel("Unique queries u_k")
    ax.set_xscale("log")
    ax.set_title("Large-k sweep: saturation at template-pool size m")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "extended_large_k.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "extended_large_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved extended sweep figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/extended")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)
    config = Config.from_yaml(args.config)

    print("==> Extended alpha sweep")
    alpha_df = run_extended_alpha_sweep(config, output_dir, args.trials)

    print("\n==> Epsilon sweep")
    eps_df = run_epsilon_sweep_validation(config, output_dir, args.trials)

    print("\n==> Large k sweep")
    k_df = run_large_k_sweep(config, output_dir, max(15, args.trials // 2))

    plot_extended(alpha_df, eps_df, k_df, output_dir)

    print("\n=== Extended sweep summary ===")
    print("Alpha sweep (k=200):")
    print(alpha_df.groupby("alpha").agg(
        pred=("predicted_unique", "first"),
        emp=("empirical_unique", "mean"),
    ).round(3).to_string())
    print("\nEpsilon sweep mean error:")
    eps_df["err"] = (eps_df["predicted_unique"] - eps_df["empirical_unique"]).abs()
    print(eps_df.groupby(["eps_q", "k"])["err"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
