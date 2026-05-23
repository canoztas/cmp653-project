"""Semantic cache on TRUE alpha-equivalent query pairs (item 5).

Section 8 of the paper reports an honest negative: semantic L2 cache misuses
queries that are merely similar but not alpha-equivalent. This experiment is
the positive complement: when the query pairs are GENUINELY equivalent (e.g.,
commutative WHERE reorder, double negation, equivalent literal forms), the
semantic L2 cache should serve them correctly with no error penalty.

We construct three equivalence classes:
  - Class A: commutative AND reordering   WHERE a AND b == WHERE b AND a
  - Class B: double negation              WHERE NOT NOT p == WHERE p
  - Class C: equivalent comparator        WHERE x >= 5 == WHERE x > 4    (integer column)

For each class, we generate paired queries and run a single workload through
WORKLOAD_DP (L1 only) and SEMANTIC_DP (L1 + L2). We expect:
  - WORKLOAD_DP: cache miss on the rephrased query => more budget spent
  - SEMANTIC_DP: cache HIT (correctly), same accuracy
"""

from __future__ import annotations

import argparse
import copy
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

sns.set_theme(style="whitegrid", font_scale=1.1)


# Three equivalence classes; each entry is a list of pairwise-equivalent SQL strings
EQUIVALENCE_CLASSES = {
    "commutative_and": [
        ("SELECT COUNT(*) FROM adult WHERE age > 30 AND education = 'Bachelors'",
         "SELECT COUNT(*) FROM adult WHERE education = 'Bachelors' AND age > 30"),
        ("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R' AND l_shipmode = 'AIR'",
         "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'AIR' AND l_returnflag = 'R'"),
    ],
    "integer_equivalent": [
        # On INTEGER columns these are equivalent:
        ("SELECT COUNT(*) FROM adult WHERE age >= 30",
         "SELECT COUNT(*) FROM adult WHERE age > 29"),
    ],
    "redundant_predicate": [
        # WHERE TRUE AND p == WHERE p — sqlglot may or may not normalize this
        ("SELECT COUNT(*) FROM adult WHERE age > 40",
         "SELECT COUNT(*) FROM adult WHERE age > 40 AND 1 = 1"),
    ],
}


def run_pair(config, mode, sql_a, sql_b, eps_q):
    """Issue (a, b) in sequence and report cache behaviour + accuracy."""
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = 1000.0
    mw = DPMiddleware(cfg, mode=mode)
    mw_exact = DPMiddleware(cfg, mode=ExecutionMode.EXACT, db=mw.db)
    true_a = float(mw_exact.execute(sql_a).rows[0][0])
    true_b = float(mw_exact.execute(sql_b).rows[0][0])
    r_a = mw.execute(sql_a, epsilon=eps_q)
    r_b = mw.execute(sql_b, epsilon=eps_q)
    noisy_a = float(r_a.rows[0][0]) if r_a.rows else float("nan")
    noisy_b = float(r_b.rows[0][0]) if r_b.rows else float("nan")
    s = mw.budget_summary() or {}
    return {
        "true_a": true_a, "true_b": true_b,
        "noisy_a": noisy_a, "noisy_b": noisy_b,
        "true_equal": true_a == true_b,
        "cache_hit_b": r_b.cache_hit,
        "consumed_eps": s.get("consumed_epsilon", 0.0),
        "abs_err_a": abs(noisy_a - true_a),
        "abs_err_b": abs(noisy_b - true_b),
    }


def run_experiment(config, output_dir, n_trials=20):
    modes = [ExecutionMode.WORKLOAD_DP, ExecutionMode.SEMANTIC_DP]
    eps_q = 1.0

    records = []
    print("[Semantic alpha-equivalent] running pairs through L1 vs L1+L2")
    for cls, pairs in EQUIVALENCE_CLASSES.items():
        for pair_idx, (sql_a, sql_b) in enumerate(pairs):
            for trial in range(n_trials):
                seed = trial * 23 + pair_idx
                np.random.seed(seed)
                for mode in modes:
                    r = run_pair(config, mode, sql_a, sql_b, eps_q)
                    r.update({
                        "class": cls, "pair_idx": pair_idx,
                        "mode": mode.value, "trial": trial,
                    })
                    records.append(r)
            print(f"  {cls} pair {pair_idx}: done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "semantic_alpha_equiv.csv", index=False)
    return df


def plot_results(df, output_dir):
    palette = {"workload_dp": "#3498db", "semantic_dp": "#2ecc71"}

    # Group by class + mode
    g = df.groupby(["class", "mode"]).agg(
        cache_hit_rate=("cache_hit_b", "mean"),
        consumed=("consumed_eps", "mean"),
        err_b=("abs_err_b", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(data=g, x="class", y="cache_hit_rate", hue="mode",
                palette=palette, ax=axes[0])
    axes[0].set_title("Cache-hit rate on the equivalent rephrasing of A")
    axes[0].set_ylim(0, 1.05)
    sns.barplot(data=g, x="class", y="consumed", hue="mode",
                palette=palette, ax=axes[1])
    axes[1].set_title("Total eps consumed on (A, B) pair")
    sns.barplot(data=g, x="class", y="err_b", hue="mode",
                palette=palette, ax=axes[2])
    axes[2].set_title("Abs error on B (the rephrased query)")
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_alpha_equiv.pdf", dpi=150)
    plt.savefig(output_dir / "semantic_alpha_equiv.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/semantic_alpha_equiv")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    df = run_experiment(config, Path(args.output), args.trials)
    plot_results(df, Path(args.output))

    print("\n=== Semantic alpha-equivalent summary ===")
    print(df.groupby(["class", "mode"]).agg(
        hits=("cache_hit_b", "mean"),
        eps=("consumed_eps", "mean"),
        err_b=("abs_err_b", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
