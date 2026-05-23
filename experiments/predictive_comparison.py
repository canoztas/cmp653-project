"""Compare predictive allocator vs fixed-eps_q workload-aware vs naive DP.

For each Zipf workload setting, run all three modes with the SAME total budget
and measure: utility, queries answered, budget exhaustion behavior. The
predictive allocator should match or beat workload-aware on utility-per-budget
because it adapts eps_q to the workload's actual occupancy.
"""

from __future__ import annotations

import argparse
import copy
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
    generate_repetitive_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_trial(config, mode, queries, eps_q_fixed, total_eps, k_total, seed):
    """Run one workload through one mode and collect metrics."""
    np.random.seed(seed)
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = total_eps

    if mode == ExecutionMode.PREDICTIVE_DP:
        mw = DPMiddleware(
            cfg, mode=mode,
            predictive_k_total=k_total,
            predictive_warmup_fraction=0.03,  # 3% warmup is enough for 7-template pool
        )
    else:
        mw = DPMiddleware(cfg, mode=mode)
    mw_exact = DPMiddleware(cfg, mode=ExecutionMode.EXACT, db=mw.db)

    abs_errors = []
    answered = 0
    eps_per_q_used = []
    t0 = time.perf_counter()
    for sql in queries:
        try:
            t_q = mw_exact.execute(sql)
            true_v = float(t_q.rows[0][0]) if t_q.rows else 0.0
            r = mw.execute(sql, epsilon=eps_q_fixed)
            if r.error is None and r.rows:
                v = float(r.rows[0][0])
                abs_errors.append(abs(v - true_v))
                eps_per_q_used.append(r.epsilon_used)
                answered += 1
        except Exception:
            pass
    wall = time.perf_counter() - t0
    s = mw.budget_summary() or {"consumed_epsilon": 0, "cache_hits": 0}

    return {
        "mode": mode.value,
        "consumed_eps": s["consumed_epsilon"],
        "cache_hits": s.get("cache_hits", 0),
        "n_answered": answered,
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "p95_abs_error": float(np.percentile(abs_errors, 95)) if abs_errors else float("nan"),
        "mean_eps_per_query": float(np.mean(eps_per_q_used)) if eps_per_q_used else 0.0,
        "wall_time_sec": wall,
    }


def run_comparison(config, output_dir, n_trials=30):
    """Sweep alpha; compare predictive vs fixed vs naive at fixed total budget."""
    alphas = [0.0, 0.5, 1.0, 2.0]
    k = 100
    total_eps = 10.0           # fixed total budget across modes
    eps_q_fixed = total_eps / k  # naive baseline divides budget evenly

    modes = [
        ExecutionMode.NAIVE_DP,
        ExecutionMode.WORKLOAD_DP,
        ExecutionMode.PREDICTIVE_DP,
    ]

    records = []
    print(f"Predictive comparison: alphas={alphas}, k={k}, total_eps={total_eps}, "
          f"fixed eps_q={eps_q_fixed:.3f}, n_trials={n_trials}")
    for alpha in alphas:
        for trial in range(n_trials):
            seed = trial * 53 + int(alpha * 1000)
            queries, _ = generate_zipf_workload(
                ADULT_AGE_BAND_TEMPLATES, alpha=alpha, k=k, seed=seed
            )
            for mode in modes:
                r = run_trial(config, mode, queries, eps_q_fixed,
                              total_eps, k, seed)
                r["alpha"] = alpha
                r["trial"] = trial
                r["k"] = k
                r["total_eps"] = total_eps
                records.append(r)
        print(f"  alpha={alpha}: {n_trials} trials done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictive_comparison.csv", index=False)
    return df


def plot_comparison(df, output_dir):
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db", "predictive_dp": "#9b59b6"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    g = df.groupby(["alpha", "mode"]).agg(
        mae=("mean_abs_error", "mean"),
        std=("mean_abs_error", "std"),
        n=("mean_abs_error", "size"),
    ).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["n"])

    sns.barplot(data=g, x="alpha", y="mae", hue="mode",
                palette=palette, ax=axes[0])
    axes[0].set_title("Mean absolute error per query (lower = better)")
    axes[0].set_xlabel(r"Zipf $\alpha$")
    axes[0].set_ylabel("MAE")

    sns.barplot(data=df, x="alpha", y="n_answered", hue="mode",
                palette=palette, ax=axes[1])
    axes[1].set_title("Queries answered within budget (out of 100)")
    axes[1].set_xlabel(r"Zipf $\alpha$")
    axes[1].set_ylabel("# answered")

    plt.tight_layout()
    plt.savefig(output_dir / "predictive_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "predictive_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/predictive")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    df = run_comparison(config, Path(args.output), args.trials)
    plot_comparison(df, Path(args.output))

    print("\n=== Predictive vs WORKLOAD vs NAIVE ===")
    print(df.groupby(["alpha", "mode"]).agg(
        eps=("consumed_eps", "mean"),
        mae=("mean_abs_error", "mean"),
        mean_eps_q=("mean_eps_per_query", "mean"),
        ans=("n_answered", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
