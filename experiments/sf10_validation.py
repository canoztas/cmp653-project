"""Cross-scale validation: does the model still hold at SF=10?

Runs the same Zipf model validation against a TPC-H SF=10 database
(10x larger than SF=1, 60M lineitem rows). Compares predicted vs empirical
unique queries to confirm that the analytical model is independent of
data scale.
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
from dpdb.model import expected_unique_queries, zipf_distribution
from dpdb.workload_gen import (
    TPCH_RETURNFLAG_TEMPLATES,
    ORDERS_PRIORITY_TEMPLATES,
    generate_zipf_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_one(config, queries, eps_q, seed):
    np.random.seed(seed)
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = 1000.0
    mw = DPMiddleware(cfg, mode=ExecutionMode.WORKLOAD_DP)
    for sql in queries:
        try:
            mw.execute(sql, epsilon=eps_q)
        except Exception:
            pass
    s = mw.budget_summary()
    return s["consumed_epsilon"] / eps_q


def cross_scale_validation(output_dir, n_trials=30):
    """Compare model predictions vs empirical at SF=1 and SF=10."""
    scales = [
        ("SF1", "data/dpdb.duckdb"),
        ("SF10", "data/dpdb_sf10.duckdb"),
    ]
    template_sets = [
        ("returnflag (m=3)", TPCH_RETURNFLAG_TEMPLATES),
        ("priority (m=5)", ORDERS_PRIORITY_TEMPLATES),
    ]
    alphas = [0.0, 0.5, 1.0, 2.0]
    k = 100
    eps_q = 1.0

    records = []
    for scale_label, db_path in scales:
        if not Path(db_path).exists():
            print(f"  Skipping {scale_label}: {db_path} not found")
            continue
        cfg = Config.from_yaml("config.yaml")
        cfg.duckdb_path = db_path

        for set_label, templates in template_sets:
            m = len(templates)
            for alpha in alphas:
                p = zipf_distribution(m, alpha)
                pred = expected_unique_queries(p, k)
                for trial in range(n_trials):
                    seed = trial * 71 + int(alpha * 100) + hash(scale_label + set_label) % 1000
                    queries, _ = generate_zipf_workload(templates, alpha=alpha,
                                                        k=k, seed=seed)
                    emp = run_one(cfg, queries, eps_q, seed)
                    records.append({
                        "scale_factor": scale_label,
                        "template_set": set_label,
                        "alpha": alpha,
                        "k": k,
                        "trial": trial,
                        "predicted_unique": pred,
                        "empirical_unique": emp,
                    })
                print(f"  {scale_label} | {set_label} | alpha={alpha}: pred={pred:.2f}")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sf10_validation.csv", index=False)
    return df


def plot_sf10(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    template_sets = sorted(df["template_set"].unique())
    for i, set_label in enumerate(template_sets):
        ax = axes[i]
        sub = df[df["template_set"] == set_label]
        for scale in sorted(sub["scale_factor"].unique()):
            ss = sub[sub["scale_factor"] == scale]
            g = ss.groupby("alpha").agg(
                emp=("empirical_unique", "mean"),
                std=("empirical_unique", "std"),
                pred=("predicted_unique", "first"),
                n=("empirical_unique", "size"),
            ).reset_index()
            ci = 1.96 * g["std"] / np.sqrt(g["n"])
            ax.errorbar(g["alpha"], g["emp"], yerr=ci, fmt="o",
                        markersize=8, label=f"{scale} empirical")
            if scale == sorted(sub["scale_factor"].unique())[0]:
                ax.plot(g["alpha"], g["pred"], "k--", linewidth=2,
                        label="Model (scale-independent)")
        ax.set_xlabel(r"Zipf $\alpha$")
        ax.set_ylabel("Unique queries u_k")
        ax.set_title(f"{set_label} — model holds across scales")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sf10_validation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "sf10_validation.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/sf10")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    df = cross_scale_validation(Path(args.output), args.trials)
    plot_sf10(df, Path(args.output))
    print("\n=== SF cross-validation summary ===")
    print(df.groupby(["scale_factor", "template_set", "alpha"]).agg(
        pred=("predicted_unique", "first"),
        emp=("empirical_unique", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
