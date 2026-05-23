"""Mixed-workload experiment (item 4): interleaved repetitive + drill-down.

Models a real dashboard that alternates between
  - W1-style auto-refresh (high-repetition dashboard tile)
  - W4-style drill-down (analyst investigating)

The model should still predict budget consumption from the workload's
effective distribution. We measure whether the prediction holds when the
template distribution is heterogeneous within a single workload.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
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
from dpdb.model import expected_unique_queries
from dpdb.parser import parse_query
from dpdb.template import extract_template, template_hash

sns.set_theme(style="whitegrid", font_scale=1.1)


def build_mixed_workload(k: int, repetition_fraction: float, seed: int):
    """k queries: repetition_fraction are W1-style refresh on a single template;
    the rest are unique W4-style drill-down queries.

    Returns: queries, true_E[u_k] (computable from construction).
    """
    rng = np.random.default_rng(seed)
    n_repeat = int(round(repetition_fraction * k))
    n_unique = k - n_repeat

    # The repeating template is one fixed COUNT(*) query
    repeat_query = "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'"

    # Drill-down: each strictly different
    flags = ["A", "N"]  # different from repeat to avoid accidental match
    modes = ["AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"]
    drill_queries = []
    for i in range(n_unique):
        flag = flags[i % 2]
        mode = modes[(i // 2) % 7]
        qty = 5 + (i // 14)
        drill_queries.append(
            f"SELECT COUNT(*) FROM lineitem WHERE l_returnflag = '{flag}' "
            f"AND l_shipmode = '{mode}' AND l_quantity > {qty}"
        )

    # Interleave: random shuffle
    all_queries = [repeat_query] * n_repeat + drill_queries
    rng.shuffle(all_queries)

    # True u_k = 1 (the repeat template) + n_unique (each drill-down distinct)
    true_u_k = 1 + n_unique
    return all_queries, true_u_k


def run_trial(config, mode, queries, eps_q_fixed, total_eps, k_total, seed):
    np.random.seed(seed)
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = total_eps
    if mode == ExecutionMode.PREDICTIVE_DP:
        mw = DPMiddleware(cfg, mode=mode, predictive_k_total=k_total,
                          predictive_warmup_fraction=0.05)
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


def run_mixed(config, output_dir, n_trials=20):
    """Sweep the fraction of repetitive queries: 0.1, 0.3, 0.5, 0.7, 0.9."""
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    k = 100
    B = 50.0
    eps_q_fixed = B / k
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP, ExecutionMode.PREDICTIVE_DP]

    records = []
    print(f"[Mixed workload] repetition fractions in {fractions}")
    for frac in fractions:
        for trial in range(n_trials):
            seed = trial * 17 + int(frac * 100)
            queries, true_u_k = build_mixed_workload(k, frac, seed)
            for mode in modes:
                r = run_trial(config, mode, queries, eps_q_fixed, B, k, seed)
                r.update({
                    "mode": mode.value, "repetition_fraction": frac,
                    "trial": trial, "true_u_k": true_u_k,
                })
                records.append(r)
        print(f"  frac={frac}: done (true u_k = {true_u_k})")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "mixed_workload.csv", index=False)
    return df


def plot_mixed(df, output_dir):
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db",
               "predictive_dp": "#9b59b6"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(data=df, x="repetition_fraction", y="consumed_eps", hue="mode",
                palette=palette, ax=axes[0])
    axes[0].set_title("Budget consumed vs repetition mix")
    axes[0].set_xlabel("Fraction of repetitive queries")
    axes[0].set_ylabel("Total eps consumed")
    sns.barplot(data=df, x="repetition_fraction", y="mean_abs_error", hue="mode",
                palette=palette, ax=axes[1])
    axes[1].set_title("MAE vs repetition mix")
    axes[1].set_xlabel("Fraction of repetitive queries")
    sns.barplot(data=df, x="repetition_fraction", y="n_answered", hue="mode",
                palette=palette, ax=axes[2])
    axes[2].set_title("Queries answered vs repetition mix")
    axes[2].set_xlabel("Fraction of repetitive queries")
    plt.tight_layout()
    plt.savefig(output_dir / "mixed_workload.pdf", dpi=150)
    plt.savefig(output_dir / "mixed_workload.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/mixed")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    df = run_mixed(config, Path(args.output), args.trials)
    plot_mixed(df, Path(args.output))

    print("\n=== Mixed-workload summary ===")
    print(df.groupby(["repetition_fraction", "mode"]).agg(
        eps=("consumed_eps", "mean"),
        ans=("n_answered", "mean"),
        mae=("mean_abs_error", "mean"),
        true_u=("true_u_k", "first"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
