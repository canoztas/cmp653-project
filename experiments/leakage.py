"""Privacy leakage experiments: membership inference + reconstruction.

This module implements R4 from the revision plan. For each privacy parameter,
we measure how well an adversary can:

(a) Decide whether a target record is in the database (membership inference)
    using the difference between query outputs on D and D \\ {target}.
(b) Reconstruct an unknown count (e.g., the count of patients with HIV in
    a small subpopulation) from a sequence of overlapping aggregate queries
    — the classical differencing/reconstruction attack pattern.

We expect:
  - As epsilon decreases, MIA AUC degrades toward 0.5 (random guessing).
  - Reconstruction error grows roughly as 1/epsilon (Laplace scale).
  - Both should align with the analytical predictions of model.py.
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

from dpdb.mechanisms import laplace_mechanism

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Membership inference attack
# ---------------------------------------------------------------------------

def mia_threshold_attack(
    n_with: int,
    n_without: int,
    eps: float,
    sensitivity: float = 1.0,
    n_trials: int = 2000,
    seed: int = 0,
) -> dict:
    """Simulate a membership inference attack on a COUNT query.

    The adversary observes a single noisy count and must decide which world
    (D with target | D without target) generated it. The optimal classifier
    is a likelihood-ratio threshold at the midpoint.

    Returns: dict with AUC, accuracy, theoretical advantage bound (exp(eps)-1).
    """
    rng = np.random.default_rng(seed)
    scale = sensitivity / eps

    # Sample noisy counts under each world
    noisy_with = n_with + rng.laplace(0.0, scale, size=n_trials)
    noisy_without = n_without + rng.laplace(0.0, scale, size=n_trials)

    # Adversary's classifier: threshold at midpoint between n_with and n_without
    threshold = 0.5 * (n_with + n_without)
    # If n_with > n_without, observation > threshold => guess "with"
    if n_with > n_without:
        pred_with = noisy_with > threshold
        pred_without = noisy_without > threshold
    else:
        pred_with = noisy_with < threshold
        pred_without = noisy_without < threshold

    tp = pred_with.sum()
    tn = (~pred_without).sum()
    fp = pred_without.sum()
    fn = (~pred_with).sum()

    accuracy = (tp + tn) / (2 * n_trials)
    # Sweep threshold for ROC -> AUC
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([noisy_with, noisy_without])
    labels = np.concatenate([np.ones(n_trials), np.zeros(n_trials)])
    # The "with" world has higher counts (since n_with > n_without by 1)
    if n_with < n_without:
        scores = -scores
    auc = roc_auc_score(labels, scores)

    # Theoretical bound on advantage: epsilon-DP gives bound exp(eps)/(1+exp(eps))
    # for optimal advantage in the bounded-difference regime
    theoretical_max_acc = np.exp(eps) / (1.0 + np.exp(eps))
    return {
        "eps": eps,
        "sensitivity": sensitivity,
        "n_with": n_with,
        "n_without": n_without,
        "mia_auc": auc,
        "mia_accuracy": accuracy,
        "theoretical_max_acc": theoretical_max_acc,
    }


def run_mia_sweep(output_dir: Path):
    """Sweep epsilon and target population size; record MIA AUC."""
    eps_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    n_with_values = [50, 500, 5000]

    records = []
    for n_with in n_with_values:
        for eps in eps_values:
            r = mia_threshold_attack(
                n_with=n_with, n_without=n_with - 1,
                eps=eps, sensitivity=1.0,
                n_trials=5000, seed=42,
            )
            records.append(r)

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "mia_results.csv", index=False)
    return df


def plot_mia(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_with in sorted(df["n_with"].unique()):
        sub = df[df["n_with"] == n_with].sort_values("eps")
        ax.plot(sub["eps"], sub["mia_auc"], "o-", label=f"n = {n_with}", linewidth=2)
    # Theoretical bound
    eps_grid = np.linspace(df["eps"].min(), df["eps"].max(), 100)
    theoretical = np.exp(eps_grid) / (1.0 + np.exp(eps_grid))
    ax.plot(eps_grid, theoretical, "k--", label=r"Theoretical bound $e^\varepsilon/(1+e^\varepsilon)$",
            linewidth=2, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", label="Random guess (AUC = 0.5)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy parameter $\varepsilon$")
    ax.set_ylabel("MIA AUC")
    ax.set_title("Membership Inference Attack on a COUNT Query (single release)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "mia_auc_vs_eps.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "mia_auc_vs_eps.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Reconstruction attack (differencing-style on drill-down)
# ---------------------------------------------------------------------------

def reconstruction_drill_down(
    true_counts: list[int],
    eps_per_query: float,
    n_trials: int = 500,
    seed: int = 0,
) -> dict:
    """Simulate the differencing-attack pattern on a sequence of nested counts.

    true_counts: a sequence of counts c_0 >= c_1 >= ... >= c_K where each c_{i+1}
                  is computed by adding one more conjunct to the WHERE clause.
                  The adversary observes noisy versions and tries to reconstruct
                  the differences c_i - c_{i+1} (each difference reveals the
                  contribution of one filter clause).

    Returns: mean and max absolute reconstruction error across trials.
    """
    rng = np.random.default_rng(seed)
    true_diffs = np.diff(true_counts)  # the secrets the adversary wants

    errors = []
    for _ in range(n_trials):
        scale = 1.0 / eps_per_query  # sensitivity = 1 for COUNT
        noisy = np.array(true_counts) + rng.laplace(0.0, scale, size=len(true_counts))
        noisy_diffs = np.diff(noisy)
        err = np.abs(noisy_diffs - true_diffs)
        errors.append(err)

    errors = np.array(errors)
    return {
        "eps_per_query": eps_per_query,
        "n_levels": len(true_counts),
        "mean_abs_error": errors.mean(),
        "max_abs_error": errors.max(),
        "p95_abs_error": np.percentile(errors, 95),
        "true_diffs": true_diffs.tolist(),
    }


def run_reconstruction_sweep(output_dir: Path):
    """Reconstruction attack on a 5-level drill-down."""
    # Example drill-down: nested predicates on adult dataset
    # c_0 = COUNT(*)
    # c_1 = COUNT WHERE age >= 30
    # c_2 = COUNT WHERE age >= 30 AND education = 'Doctorate'
    # c_3 = ... AND sex = 'Female'
    # c_4 = ... AND native_country = 'United-States'
    # Use realistic numbers from the Adult dataset (approximately)
    true_counts = [48842, 33049, 365, 99, 89]
    eps_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    records = []
    for eps in eps_values:
        r = reconstruction_drill_down(true_counts, eps_per_query=eps, n_trials=2000, seed=42)
        r["mode"] = "naive_full_budget_per_level"
        records.append(r)

        # Under workload-aware: only c_0 is "the" cached version,
        # subsequent are new queries (no overlap in cache).
        # But: budget can be spread more thinly because fewer distinct templates.
        # Approximation: with workload-aware on a 1-level repeat dashboard
        # the adversary gets eps_per_query for c_0 only.
        # On drill-down, no caching benefit. Same as naive.

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "reconstruction_results.csv", index=False)
    return df


def plot_reconstruction(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["eps_per_query"], df["mean_abs_error"], "o-", label="Mean abs error per diff", linewidth=2)
    ax.plot(df["eps_per_query"], df["p95_abs_error"], "s--", label="p95 abs error per diff", linewidth=2)
    # Theoretical: |noisy_a - noisy_b| has scale 2/eps (sum of two Laplace noises)
    eps_grid = np.logspace(np.log10(df["eps_per_query"].min()), np.log10(df["eps_per_query"].max()), 100)
    ax.plot(eps_grid, np.sqrt(2) * np.sqrt(2) / eps_grid, "k--",
            label=r"Theoretical std of $\eta_1 - \eta_2$ : $2/\varepsilon$", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Per-query privacy parameter $\varepsilon$")
    ax.set_ylabel("Reconstruction error (count units)")
    ax.set_title("Differencing Attack on Drill-Down: Reconstruction Error vs $\\varepsilon$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_vs_eps.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "reconstruction_vs_eps.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/leakage")
    args = parser.parse_args()
    output_dir = Path(args.output)

    print("[Leakage] Running MIA sweep...")
    mia_df = run_mia_sweep(output_dir)
    plot_mia(mia_df, output_dir)
    print(f"  Saved {output_dir / 'mia_results.csv'}")

    print("[Leakage] Running reconstruction sweep...")
    rec_df = run_reconstruction_sweep(output_dir)
    plot_reconstruction(rec_df, output_dir)
    print(f"  Saved {output_dir / 'reconstruction_results.csv'}")

    print("\n[Leakage] MIA summary (selected):")
    print(mia_df[mia_df["n_with"] == 500].round(4).to_string(index=False))
    print("\n[Leakage] Reconstruction summary:")
    print(rec_df[["eps_per_query", "mean_abs_error", "p95_abs_error"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
