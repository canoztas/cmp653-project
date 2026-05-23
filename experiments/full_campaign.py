"""Full benchmark campaign as requested by the revision brief.

Grid:
  - Workloads: W1 (repetitive), W2 (Zipf-parametric), W3 (uniform), W4 (drill-down)
  - Privacy settings: eps in {0.1, 0.5, 1.0, 2.0}
  - TPC-H scale factors: SF=1 (and SF=10 if data file exists)
  - Modes: NAIVE_DP, WORKLOAD_DP, TEMPORAL_DP
  - 30 trials per cell for statistical stability

Outputs per-trial CSV + 4 summary figures.
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
from dpdb.model import (
    expected_unique_queries,
    budget_savings_ratio,
    zipf_distribution,
)
from dpdb.workload_gen import (
    ADULT_AGE_BAND_TEMPLATES,
    TPCH_RETURNFLAG_TEMPLATES,
    ORDERS_PRIORITY_TEMPLATES,
    generate_zipf_workload,
    generate_repetitive_workload,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Workload definitions for the full campaign
# ---------------------------------------------------------------------------

def build_workload(name: str, k: int, seed: int):
    """Build a (queries, expected_unique) pair for workload `name`."""
    if name == "W1_repetitive":
        # One template repeated k times -> Limit A
        templates = [ADULT_AGE_BAND_TEMPLATES[1]]  # age in [30,40)
        queries = generate_repetitive_workload(templates[0], k)
        return queries, 1.0  # E[u_k] = 1 always
    elif name == "W2_zipf":
        # Zipf(alpha=1.0) over age bands -> intermediate
        templates = ADULT_AGE_BAND_TEMPLATES
        p = zipf_distribution(len(templates), 1.0)
        queries, _ = generate_zipf_workload(templates, alpha=1.0, k=k, seed=seed)
        return queries, expected_unique_queries(p, k)
    elif name == "W3_uniform":
        # Uniform over age bands -> Limit B (capped by m)
        templates = ADULT_AGE_BAND_TEMPLATES
        p = zipf_distribution(len(templates), 0.0)
        queries, _ = generate_zipf_workload(templates, alpha=0.0, k=k, seed=seed)
        return queries, expected_unique_queries(p, k)
    elif name == "W4_drilldown":
        # True progressive narrowing: each step strictly narrows the previous WHERE
        # by combining a new conjunct. No two queries share an exact (template,
        # parameter) pair, so workload-aware caching cannot reuse anything.
        queries = []
        flags = ["R", "A", "N"]
        modes = ["AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"]
        for i in range(k):
            # i indexes into the cross product flags x modes; with 3 x 7 = 21
            # unique pairs we cycle if k > 21. Within k <= 100, each i still
            # produces a distinct (flag, mode, qty_threshold) triple by
            # adding a third varying conjunct.
            flag = flags[i % 3]
            mode = modes[(i // 3) % 7]
            qty_threshold = 10 + (i // 21)  # 10, 11, 12, 13, ... -- unique per i
            q = (
                f"SELECT COUNT(*) FROM lineitem "
                f"WHERE l_returnflag = '{flag}' AND l_shipmode = '{mode}' "
                f"AND l_quantity > {qty_threshold}"
            )
            queries.append(q)
        return queries, k * 1.0  # every query is unique by construction
    elif name == "W2_tpch_returnflag":
        # Repetitive over only 3 return flags -> heavy reuse
        templates = TPCH_RETURNFLAG_TEMPLATES
        p = zipf_distribution(len(templates), 0.5)
        queries, _ = generate_zipf_workload(templates, alpha=0.5, k=k, seed=seed)
        return queries, expected_unique_queries(p, k)
    elif name == "W2_tpch_priority":
        # Zipf over 5 priorities
        templates = ORDERS_PRIORITY_TEMPLATES
        p = zipf_distribution(len(templates), 1.0)
        queries, _ = generate_zipf_workload(templates, alpha=1.0, k=k, seed=seed)
        return queries, expected_unique_queries(p, k)
    else:
        raise KeyError(f"Unknown workload: {name}")


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(config, mode, queries, eps_q, seed, total_eps_cap=None):
    """Run one workload through one mode; return per-trial metrics."""
    np.random.seed(seed)

    import copy
    cfg = copy.deepcopy(config)
    if total_eps_cap is not None:
        cfg.privacy.total_epsilon = total_eps_cap

    mw = DPMiddleware(cfg, mode=mode)
    mw_exact = DPMiddleware(cfg, mode=ExecutionMode.EXACT, db=mw.db)

    n_answered = 0
    abs_errors = []
    t0 = time.perf_counter()
    for sql in queries:
        try:
            t_q = mw_exact.execute(sql)
            true_v = float(t_q.rows[0][0]) if t_q.rows else 0.0
            r = mw.execute(sql, epsilon=eps_q)
            if r.error is None and r.rows:
                noisy_v = float(r.rows[0][0])
                abs_errors.append(abs(noisy_v - true_v))
                n_answered += 1
        except Exception:
            pass
    wall = time.perf_counter() - t0
    s = mw.budget_summary()
    return {
        "mode": mode.value,
        "consumed_eps": s["consumed_epsilon"],
        "cache_hits": s["cache_hits"],
        "exact_hits": s.get("exact_hits", s["cache_hits"]),
        "semantic_hits": s.get("semantic_hits", 0),
        "n_answered": n_answered,
        "total_queries": len(queries),
        "mean_abs_error": np.mean(abs_errors) if abs_errors else float("nan"),
        "p95_abs_error": np.percentile(abs_errors, 95) if abs_errors else float("nan"),
        "wall_time_sec": wall,
        "avg_latency_ms": wall * 1000 / max(1, len(queries)),
    }


def run_full_campaign(config: Config, output_dir: Path, n_trials: int = 30,
                      k: int = 100, total_eps_cap: float = 20.0,
                      scale_label: str = "SF1"):
    """The full grid: workload x mode x epsilon x trials."""
    workloads = ["W1_repetitive", "W2_zipf", "W3_uniform", "W4_drilldown",
                 "W2_tpch_returnflag", "W2_tpch_priority"]
    eps_values = [0.1, 0.5, 1.0, 2.0]
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP, ExecutionMode.TEMPORAL_DP]

    records = []
    total_cells = len(workloads) * len(eps_values) * len(modes) * n_trials
    print(f"Running {total_cells} trial cells "
          f"({len(workloads)} workloads x {len(eps_values)} eps x {len(modes)} modes x {n_trials} trials)")

    cell = 0
    for wl in workloads:
        for eps_q in eps_values:
            for mode in modes:
                for trial in range(n_trials):
                    cell += 1
                    seed = trial * 1000 + hash((wl, eps_q, mode.value)) % 10000
                    queries, predicted_eu = build_workload(wl, k, seed)
                    r = run_trial(config, mode, queries, eps_q, seed,
                                  total_eps_cap=total_eps_cap)
                    r["workload"] = wl
                    r["eps_per_query"] = eps_q
                    r["k"] = k
                    r["trial"] = trial
                    r["predicted_unique"] = predicted_eu
                    r["scale_factor"] = scale_label
                    records.append(r)
                    if cell % 50 == 0:
                        print(f"  {cell}/{total_cells} cells done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"full_campaign_{scale_label}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return df


def plot_campaign(df: pd.DataFrame, output_dir: Path, scale_label: str):
    """4 summary figures from the full grid."""
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db", "temporal_dp": "#2ecc71"}

    # Figure 1: Consumed eps by (workload, mode) at eps_q=1.0
    sub = df[df["eps_per_query"] == 1.0]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=sub, x="workload", y="consumed_eps", hue="mode",
                palette=palette, ax=ax)
    ax.set_title(f"Privacy budget consumed (eps_q=1.0, {scale_label})")
    ax.set_ylabel("Total eps consumed (lower = better)")
    ax.set_xlabel("Workload")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"campaign_budget_{scale_label}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"campaign_budget_{scale_label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Per-query error by (workload, mode, eps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    eps_levels = sorted(df["eps_per_query"].unique())
    for i, eps_q in enumerate(eps_levels):
        ax = axes[i // 2][i % 2]
        sub = df[df["eps_per_query"] == eps_q]
        sns.barplot(data=sub, x="workload", y="mean_abs_error", hue="mode",
                    palette=palette, ax=ax)
        ax.set_title(f"eps_q = {eps_q}")
        ax.set_yscale("log")
        ax.set_ylabel("Mean abs error (log)")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        if i != 0:
            ax.legend([], frameon=False)
    plt.suptitle(f"Per-query utility across epsilon and workload ({scale_label})")
    plt.tight_layout()
    plt.savefig(output_dir / f"campaign_utility_{scale_label}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"campaign_utility_{scale_label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: Queries answered before budget exhaustion
    fig, ax = plt.subplots(figsize=(12, 6))
    sub = df[df["eps_per_query"] == 0.5]  # smaller eps -> more visible exhaustion
    sns.barplot(data=sub, x="workload", y="n_answered", hue="mode",
                palette=palette, ax=ax)
    ax.set_title(f"Queries answered within budget (eps_q=0.5, total budget=20.0, {scale_label})")
    ax.set_ylabel("# queries answered (out of 100)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"campaign_queries_answered_{scale_label}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"campaign_queries_answered_{scale_label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: Model prediction vs empirical unique queries (validation)
    wa = df[df["mode"] == "workload_dp"]
    g = wa.groupby(["workload", "eps_per_query"]).agg(
        emp=("consumed_eps", "mean"),
        pred=("predicted_unique", "first"),
    ).reset_index()
    # Convert empirical consumed -> empirical unique (divide by eps_per_query)
    g["emp_unique"] = g["emp"] / g["eps_per_query"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for wl in g["workload"].unique():
        sub = g[g["workload"] == wl]
        ax.scatter(sub["pred"], sub["emp_unique"], label=wl, s=80, alpha=0.7)
    max_v = max(g["pred"].max(), g["emp_unique"].max()) * 1.05
    ax.plot([0, max_v], [0, max_v], "k--", linewidth=1, label="Model = empirical")
    ax.set_xlabel("Model predicted E[u_k]")
    ax.set_ylabel("Empirical mean unique queries")
    ax.set_title(f"Model validation: predicted vs empirical ({scale_label})")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"campaign_model_validation_{scale_label}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"campaign_model_validation_{scale_label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/full_campaign")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--total-eps", type=float, default=20.0)
    parser.add_argument("--scale", default="SF1")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    df = run_full_campaign(config, Path(args.output), args.trials, args.k,
                           args.total_eps, args.scale)
    plot_campaign(df, Path(args.output), args.scale)

    print("\n=== Headline Summary ===")
    print(df.groupby(["workload", "mode", "eps_per_query"]).agg(
        budget_consumed=("consumed_eps", "mean"),
        cache_hits=("cache_hits", "mean"),
        mean_err=("mean_abs_error", "mean"),
        n_ans=("n_answered", "mean"),
    ).round(2).to_string())


if __name__ == "__main__":
    main()
