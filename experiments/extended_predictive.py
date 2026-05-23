"""Three follow-up sweeps for the predictive allocator (items 1, 2, 6).

1. Budget sweep: B in {1, 5, 10, 50}. Shows how the predictive vs naive gap
   depends on the total budget regime.
2. Cross-scale: predictive at SF=10 to confirm scale-invariance.
3. Predictive + Temporal: combined with staleness tolerance to show that
   the two new mechanisms compose.
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
    ADULT_AGE_BAND_TEMPLATES, TPCH_RETURNFLAG_TEMPLATES,
    generate_zipf_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_one(config, mode, queries, eps_q_fixed, total_eps, k_total, seed,
            staleness=float("inf")):
    np.random.seed(seed)
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = total_eps
    if mode == ExecutionMode.PREDICTIVE_DP:
        mw = DPMiddleware(cfg, mode=mode, predictive_k_total=k_total,
                          predictive_warmup_fraction=0.03,
                          staleness_tolerance=staleness)
    elif mode == ExecutionMode.TEMPORAL_DP:
        mw = DPMiddleware(cfg, mode=mode, staleness_tolerance=staleness)
    else:
        mw = DPMiddleware(cfg, mode=mode)
    mw_exact = DPMiddleware(cfg, mode=ExecutionMode.EXACT, db=mw.db)
    abs_errors = []
    answered = 0
    for sql in queries:
        try:
            t_q = mw_exact.execute(sql)
            true_v = float(t_q.rows[0][0]) if t_q.rows else 0.0
            r = mw.execute(sql, epsilon=eps_q_fixed)
            if r.error is None and r.rows:
                abs_errors.append(abs(float(r.rows[0][0]) - true_v))
                answered += 1
        except Exception:
            pass
    s = mw.budget_summary() or {}
    return {
        "consumed_eps": s.get("consumed_epsilon", 0.0),
        "cache_hits": s.get("cache_hits", 0),
        "n_answered": answered,
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else float("nan"),
    }


def budget_sweep(config, output_dir, n_trials=20):
    """Predictive vs Naive vs Workload across budget levels B in {1, 5, 10, 50}."""
    budgets = [1.0, 5.0, 10.0, 50.0]
    alpha = 1.0
    k = 100
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP, ExecutionMode.PREDICTIVE_DP]

    records = []
    print("[Budget sweep] B in {1,5,10,50}, alpha=1.0, k=100")
    for B in budgets:
        eps_q_fixed = B / k  # naive split for the baselines
        for trial in range(n_trials):
            seed = trial * 31 + int(B * 100)
            queries, _ = generate_zipf_workload(
                ADULT_AGE_BAND_TEMPLATES, alpha=alpha, k=k, seed=seed
            )
            for mode in modes:
                r = run_one(config, mode, queries, eps_q_fixed, B, k, seed)
                r.update({"mode": mode.value, "total_budget": B, "trial": trial})
                records.append(r)
        print(f"  B={B}: done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictive_budget_sweep.csv", index=False)
    return df


def sf10_predictive(output_dir, n_trials=20):
    """Run predictive vs baselines on SF=10 (cross-scale validation)."""
    cfg_sf10 = Config.from_yaml("config.yaml")
    cfg_sf10.duckdb_path = "data/dpdb_sf10.duckdb"
    if not Path(cfg_sf10.duckdb_path).exists():
        print(f"  Skipping SF=10: {cfg_sf10.duckdb_path} not found")
        return None

    alpha = 1.0
    k = 100
    B = 10.0
    eps_q_fixed = B / k
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP, ExecutionMode.PREDICTIVE_DP]

    records = []
    print("[SF=10 predictive] alpha=1, k=100, B=10")
    for trial in range(n_trials):
        seed = trial * 43
        queries, _ = generate_zipf_workload(
            TPCH_RETURNFLAG_TEMPLATES, alpha=alpha, k=k, seed=seed
        )
        for mode in modes:
            r = run_one(cfg_sf10, mode, queries, eps_q_fixed, B, k, seed)
            r.update({"mode": mode.value, "scale": "SF10", "trial": trial})
            records.append(r)
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "predictive_sf10.csv", index=False)
    return df


def predictive_x_temporal(config, output_dir, n_trials=20):
    """Predictive allocator combined with temporal staleness tolerance."""
    tau_values = [10, 25, 50, 100, 1000]  # staleness tolerance in query steps
    alpha = 1.0
    k = 100
    B = 50.0  # enough budget so staleness drives re-noising

    records = []
    print("[Predictive x Temporal] tau in {10,25,50,100,1000}")
    for tau in tau_values:
        for trial in range(n_trials):
            seed = trial * 53 + tau
            queries, _ = generate_zipf_workload(
                ADULT_AGE_BAND_TEMPLATES, alpha=alpha, k=k, seed=seed
            )
            # Predictive with staleness tolerance
            r_pred = run_one(config, ExecutionMode.PREDICTIVE_DP, queries,
                             B / k, B, k, seed, staleness=tau)
            r_pred.update({"mode": "predictive_dp", "tau": tau, "trial": trial})
            records.append(r_pred)
            # Temporal alone (no predictive)
            r_temp = run_one(config, ExecutionMode.TEMPORAL_DP, queries,
                             B / k, B, k, seed, staleness=tau)
            r_temp.update({"mode": "temporal_dp", "tau": tau, "trial": trial})
            records.append(r_temp)
        print(f"  tau={tau}: done")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "predictive_x_temporal.csv", index=False)
    return df


def plot_all(budget_df, sf10_df, temporal_df, output_dir):
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db",
               "predictive_dp": "#9b59b6", "temporal_dp": "#2ecc71"}

    # 1) Budget sweep figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    g = budget_df.groupby(["total_budget", "mode"]).agg(
        mae=("mean_abs_error", "mean"),
        n_ans=("n_answered", "mean"),
        eps=("consumed_eps", "mean"),
    ).reset_index()
    sns.barplot(data=g, x="total_budget", y="mae", hue="mode",
                palette=palette, ax=axes[0])
    axes[0].set_title("MAE across budget regimes (k=100, alpha=1)")
    axes[0].set_xlabel("Total budget B")
    axes[0].set_ylabel("Mean abs error per query")
    sns.barplot(data=g, x="total_budget", y="n_ans", hue="mode",
                palette=palette, ax=axes[1])
    axes[1].set_title("Queries answered across budget regimes")
    axes[1].set_xlabel("Total budget B")
    axes[1].set_ylabel("Queries answered (out of 100)")
    plt.tight_layout()
    plt.savefig(output_dir / "extended_predictive_budget_sweep.pdf", dpi=150)
    plt.savefig(output_dir / "extended_predictive_budget_sweep.png", dpi=150)
    plt.close()

    # 2) SF=10 figure
    if sf10_df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sf10_df, x="mode", y="mean_abs_error", palette=palette, ax=ax)
        ax.set_title("Predictive vs baselines at TPC-H SF=10 (60M lineitem)")
        ax.set_ylabel("Mean absolute error")
        plt.tight_layout()
        plt.savefig(output_dir / "extended_predictive_sf10.pdf", dpi=150)
        plt.savefig(output_dir / "extended_predictive_sf10.png", dpi=150)
        plt.close()

    # 3) Predictive x Temporal
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    g = temporal_df.groupby(["tau", "mode"]).agg(
        consumed=("consumed_eps", "mean"),
        mae=("mean_abs_error", "mean"),
    ).reset_index()
    palette2 = {"predictive_dp": "#9b59b6", "temporal_dp": "#2ecc71"}
    sns.barplot(data=g, x="tau", y="consumed", hue="mode", palette=palette2, ax=axes[0])
    axes[0].set_title("Budget consumed under staleness regime")
    axes[0].set_xlabel(r"Staleness tolerance $\tau$")
    axes[0].set_ylabel("Total eps consumed")
    sns.barplot(data=g, x="tau", y="mae", hue="mode", palette=palette2, ax=axes[1])
    axes[1].set_title("MAE under staleness regime")
    axes[1].set_xlabel(r"Staleness tolerance $\tau$")
    axes[1].set_ylabel("Mean abs error")
    plt.tight_layout()
    plt.savefig(output_dir / "extended_predictive_temporal.pdf", dpi=150)
    plt.savefig(output_dir / "extended_predictive_temporal.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/predictive_extra")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    config = Config.from_yaml(args.config)

    print("==> 1. Budget sweep")
    budget_df = budget_sweep(config, output_dir, args.trials)
    print("\n==> 2. SF=10 cross-scale")
    sf10_df = sf10_predictive(output_dir, args.trials)
    print("\n==> 3. Predictive x Temporal")
    temporal_df = predictive_x_temporal(config, output_dir, args.trials)

    plot_all(budget_df, sf10_df, temporal_df, output_dir)

    print("\n=== Budget sweep summary ===")
    print(budget_df.groupby(["total_budget", "mode"]).agg(
        mae=("mean_abs_error", "mean"),
        n_ans=("n_answered", "mean"),
        eps=("consumed_eps", "mean"),
    ).round(3).to_string())

    if sf10_df is not None:
        print("\n=== SF=10 summary ===")
        print(sf10_df.groupby("mode").agg(
            mae=("mean_abs_error", "mean"),
            eps=("consumed_eps", "mean"),
        ).round(3).to_string())

    print("\n=== Predictive x Temporal summary ===")
    print(temporal_df.groupby(["tau", "mode"]).agg(
        mae=("mean_abs_error", "mean"),
        eps=("consumed_eps", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
