"""Generate publication-quality plots from benchmark results."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {"naive_dp": "#e74c3c", "workload_dp": "#2ecc71"}


def load_results(results_dir: Path):
    results = pd.read_csv(results_dir / "results.csv")
    budgets = pd.read_csv(results_dir / "budgets.csv")
    return results, budgets


def plot_mae_by_workload(results: pd.DataFrame, output_dir: Path):
    """Bar chart: MAE per workload per mode."""
    agg = (
        results.dropna(subset=["abs_error"])
        .groupby(["workload", "mode"])["abs_error"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=agg, x="workload", y="abs_error", hue="mode",
                palette=PALETTE, ax=ax)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Workload")
    ax.set_title("Utility: MAE by Workload and Mode")
    ax.legend(title="Mode")
    plt.tight_layout()
    fig.savefig(output_dir / "mae_by_workload.pdf", dpi=150)
    fig.savefig(output_dir / "mae_by_workload.png", dpi=150)
    plt.close()


def plot_cumulative_epsilon(results: pd.DataFrame, output_dir: Path):
    """Line chart: cumulative epsilon consumed vs query index."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    workloads = sorted(results["workload"].unique())

    for idx, wl in enumerate(workloads[:4]):
        ax = axes[idx // 2][idx % 2]
        subset = results[(results["workload"] == wl) & (results["trial"] == 0)]

        for mode in ["naive_dp", "workload_dp"]:
            mode_data = subset[subset["mode"] == mode]
            # Get per-query epsilon (deduplicate by query_idx)
            per_query = mode_data.drop_duplicates("query_idx")[["query_idx", "epsilon_used"]]
            per_query = per_query.sort_values("query_idx")
            cumulative = per_query["epsilon_used"].cumsum()
            ax.plot(per_query["query_idx"], cumulative,
                    label=mode, color=PALETTE[mode], linewidth=2)

        ax.set_title(wl)
        ax.set_xlabel("Query Index")
        ax.set_ylabel("Cumulative Epsilon")
        ax.legend()

    plt.suptitle("Privacy Budget Consumption", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_epsilon.pdf", dpi=150)
    fig.savefig(output_dir / "cumulative_epsilon.png", dpi=150)
    plt.close()


def plot_cache_hit_rate(budgets: pd.DataFrame, output_dir: Path):
    """Bar chart: cache hit rate per workload (workload_dp only)."""
    wa = budgets[budgets["strategy"] == "workload_aware"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=wa, x="workload", y="cache_hit_rate", color="#2ecc71", ax=ax)
    ax.set_ylabel("Cache Hit Rate")
    ax.set_xlabel("Workload")
    ax.set_title("Workload-Aware: Cache Hit Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_dir / "cache_hit_rate.pdf", dpi=150)
    fig.savefig(output_dir / "cache_hit_rate.png", dpi=150)
    plt.close()


def plot_latency_comparison(results: pd.DataFrame, output_dir: Path):
    """Box plot: latency distribution per mode per workload."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=results.drop_duplicates(["workload", "mode", "trial", "query_idx"]),
        x="workload", y="latency_ms", hue="mode",
        palette=PALETTE, ax=ax,
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Workload")
    ax.set_title("Query Latency Distribution")
    plt.tight_layout()
    fig.savefig(output_dir / "latency_comparison.pdf", dpi=150)
    fig.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close()


def plot_relative_error_distribution(results: pd.DataFrame, output_dir: Path):
    """Violin plot: relative error distribution."""
    valid = results.dropna(subset=["rel_error"])
    valid = valid[valid["rel_error"] < 10]  # cap outliers for visibility
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=valid, x="workload", y="rel_error", hue="mode",
                   palette=PALETTE, split=True, ax=ax, inner="quartile")
    ax.set_ylabel("Relative Error")
    ax.set_xlabel("Workload")
    ax.set_title("Relative Error Distribution")
    plt.tight_layout()
    fig.savefig(output_dir / "relative_error_dist.pdf", dpi=150)
    fig.savefig(output_dir / "relative_error_dist.png", dpi=150)
    plt.close()


def plot_budget_efficiency(budgets: pd.DataFrame, output_dir: Path):
    """Bar chart: total queries answered and epsilon consumed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=budgets, x="workload", y="consumed_epsilon", hue="strategy",
                palette={"naive": "#e74c3c", "workload_aware": "#2ecc71"}, ax=axes[0])
    axes[0].set_title("Total Epsilon Consumed")
    axes[0].set_ylabel("Epsilon")

    sns.barplot(data=budgets, x="workload", y="total_queries", hue="strategy",
                palette={"naive": "#e74c3c", "workload_aware": "#2ecc71"}, ax=axes[1])
    axes[1].set_title("Queries Answered")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(output_dir / "budget_efficiency.pdf", dpi=150)
    fig.savefig(output_dir / "budget_efficiency.png", dpi=150)
    plt.close()


def generate_all_plots(results_dir: Path):
    results, budgets = load_results(results_dir)
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    plot_mae_by_workload(results, output_dir)
    plot_cumulative_epsilon(results, output_dir)
    plot_cache_hit_rate(budgets, output_dir)
    plot_latency_comparison(results, output_dir)
    plot_relative_error_distribution(results, output_dir)
    plot_budget_efficiency(budgets, output_dir)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Results directory")
    args = parser.parse_args()
    generate_all_plots(Path(args.results))
