"""Empirical validation of the R3 temporal extension.

For each (staleness tolerance tau, update rate lambda) cell, we:
  1. Compute analytical prediction from model.py Proposition 6
  2. Run N trials with the TEMPORAL_DP middleware over a longer workload
  3. Compare empirical budget vs analytical prediction
"""

from __future__ import annotations

import argparse
import os
import sys
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
from dpdb.model import (
    TemporalRegime,
    expected_budget_temporal,
    expected_unique_queries,
    zipf_distribution,
)
from dpdb.workload_gen import ADULT_AGE_BAND_TEMPLATES, generate_zipf_workload

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_temporal_trial(
    config: Config,
    queries: list[str],
    eps_per_query: float,
    staleness_tolerance: float,
    update_rate: float,
    update_invalidation_prob: float,
    seed: int,
) -> dict:
    np.random.seed(seed)
    # Use a much larger budget for temporal experiment so the cap doesn't
    # mask the comparison with the model prediction.
    import copy
    big_config = copy.deepcopy(config)
    big_config.privacy.total_epsilon = 1000.0
    mw = DPMiddleware(
        big_config, mode=ExecutionMode.TEMPORAL_DP,
        staleness_tolerance=staleness_tolerance,
        update_rate=update_rate,
        update_invalidation_prob=update_invalidation_prob,
        update_seed=seed,
    )
    for sql in queries:
        try:
            mw.execute(sql, epsilon=eps_per_query)
        except Exception:
            pass
    s = mw.budget_summary()
    return {
        "consumed_epsilon": s["consumed_epsilon"],
        "cache_hits": s["cache_hits"],
        "expired": s.get("expired_evictions", 0),
        "update_evicted": s.get("update_evictions", 0),
    }


def run_temporal_validation(config: Config, output_dir: Path, eps_per_query: float = 1.0,
                            n_trials: int = 10):
    """Sweep over staleness tolerance tau and update rate lambda."""
    k = 100
    alpha = 1.0
    templates = ADULT_AGE_BAND_TEMPLATES
    m = len(templates)
    p = zipf_distribution(m, alpha)
    eu = expected_unique_queries(p, k)

    # Sweep tau (no updates)
    tau_values = [10, 25, 50, 100, 1000]
    records = []
    for tau in tau_values:
        regime = TemporalRegime(horizon_T=k, staleness_tolerance=tau)
        predicted = expected_budget_temporal(p, k_total=k, eps_q=eps_per_query, regime=regime)
        for trial in range(n_trials):
            seed = trial * 17 + tau
            queries, _ = generate_zipf_workload(templates, alpha=alpha, k=k, seed=seed)
            r = run_temporal_trial(config, queries, eps_per_query,
                                   staleness_tolerance=tau, update_rate=0.0,
                                   update_invalidation_prob=0.0, seed=seed)
            records.append({
                "experiment": "tau_sweep",
                "tau": tau,
                "lambda": 0.0,
                "trial": trial,
                "predicted_eps": predicted,
                "empirical_eps": r["consumed_epsilon"],
                "expired": r["expired"],
                "update_evicted": r["update_evicted"],
            })

    # Sweep lambda (fixed tau)
    lambda_values = [0.0, 0.05, 0.10, 0.20]
    tau_fixed = 1000  # essentially no staleness expiry
    for lam in lambda_values:
        regime = TemporalRegime(horizon_T=k, staleness_tolerance=tau_fixed,
                                update_rate=lam, update_invalidation_prob=0.3)
        predicted = expected_budget_temporal(p, k_total=k, eps_q=eps_per_query, regime=regime)
        for trial in range(n_trials):
            seed = trial * 23 + int(lam * 1000)
            queries, _ = generate_zipf_workload(templates, alpha=alpha, k=k, seed=seed)
            r = run_temporal_trial(config, queries, eps_per_query,
                                   staleness_tolerance=tau_fixed, update_rate=lam,
                                   update_invalidation_prob=0.3, seed=seed)
            records.append({
                "experiment": "lambda_sweep",
                "tau": tau_fixed,
                "lambda": lam,
                "trial": trial,
                "predicted_eps": predicted,
                "empirical_eps": r["consumed_epsilon"],
                "expired": r["expired"],
                "update_evicted": r["update_evicted"],
            })

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "temporal_validation.csv", index=False)
    return df


def plot_temporal(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # tau sweep
    tau_df = df[df["experiment"] == "tau_sweep"]
    g = tau_df.groupby("tau").agg(
        emp_mean=("empirical_eps", "mean"),
        emp_std=("empirical_eps", "std"),
        pred=("predicted_eps", "first"),
        n=("empirical_eps", "size"),
    ).reset_index()
    ci = 1.96 * g["emp_std"] / np.sqrt(g["n"])
    axes[0].errorbar(g["tau"], g["emp_mean"], yerr=ci, fmt="o", color="C0", markersize=8,
                     label="Empirical (mean ± 95% CI)")
    axes[0].plot(g["tau"], g["pred"], "--", color="C3", linewidth=2,
                 label="Model prediction")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Staleness tolerance $\tau$ (query steps)")
    axes[0].set_ylabel(r"$\varepsilon$ consumed")
    axes[0].set_title(r"Effect of staleness tolerance $\tau$ ($\lambda=0$)")
    axes[0].legend()

    # lambda sweep
    lam_df = df[df["experiment"] == "lambda_sweep"]
    g2 = lam_df.groupby("lambda").agg(
        emp_mean=("empirical_eps", "mean"),
        emp_std=("empirical_eps", "std"),
        pred=("predicted_eps", "first"),
        n=("empirical_eps", "size"),
    ).reset_index()
    ci2 = 1.96 * g2["emp_std"] / np.sqrt(g2["n"])
    axes[1].errorbar(g2["lambda"], g2["emp_mean"], yerr=ci2, fmt="o", color="C0", markersize=8,
                     label="Empirical (mean ± 95% CI)")
    axes[1].plot(g2["lambda"], g2["pred"], "--", color="C3", linewidth=2,
                 label="Model prediction")
    axes[1].set_xlabel(r"Update rate $\lambda$ (events per step)")
    axes[1].set_ylabel(r"$\varepsilon$ consumed")
    axes[1].set_title(r"Effect of update rate $\lambda$ ($\tau$ large)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_validation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "temporal_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal validation figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/temporal")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    df = run_temporal_validation(config, Path(args.output), args.epsilon, args.trials)
    print("\n=== Summary ===")
    print(df.groupby(["experiment", "tau", "lambda"]).agg(
        emp=("empirical_eps", "mean"),
        pred=("predicted_eps", "first"),
        expired=("expired", "mean"),
        invalidated=("update_evicted", "mean"),
    ).round(3).to_string())
    plot_temporal(df, Path(args.output))


if __name__ == "__main__":
    main()
