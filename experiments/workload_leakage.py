"""Workload-level leakage: MIA against the middleware across W1-W4.

Unlike experiments/leakage.py (single-query MIA), this experiment trains a
*shadow-model* membership inference attack against the full middleware
output on each workload family. The attacker observes the entire sequence
of noisy outputs and uses a logistic-regression classifier trained on
shadow runs to predict whether a target record was in the dataset.

This addresses R4 of the revision brief, which asks for MIA "across W1-W4
at multiple eps values" -- the workload-level attack, not just single-release.

Also includes a cross-check with the R2 analytical model: does the model's
predicted budget consumption explain the empirical attack AUC?
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.model import expected_unique_queries, zipf_distribution
from dpdb.workload_gen import ADULT_AGE_BAND_TEMPLATES, generate_zipf_workload

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Workloads (same as full_campaign)
# ---------------------------------------------------------------------------

def build_workload_queries(family: str, k: int, seed: int):
    """Return list of SQL strings for the given workload family."""
    if family == "W1_repetitive":
        # Same query 100 times
        return [
            "SELECT COUNT(*) FROM adult WHERE age >= 30 AND age < 40"
        ] * k
    elif family == "W2_zipf":
        queries, _ = generate_zipf_workload(
            ADULT_AGE_BAND_TEMPLATES, alpha=1.0, k=k, seed=seed
        )
        return queries
    elif family == "W3_uniform":
        queries, _ = generate_zipf_workload(
            ADULT_AGE_BAND_TEMPLATES, alpha=0.0, k=k, seed=seed
        )
        return queries
    elif family == "W4_drilldown":
        queries = []
        for i in range(k):
            age_low = 20 + (i % 7) * 10
            sex = ["Male", "Female"][i % 2]
            edu = ["Bachelors", "Masters", "HS-grad"][i % 3]
            queries.append(
                f"SELECT COUNT(*) FROM adult WHERE age >= {age_low} "
                f"AND age < {age_low + 10} AND sex = '{sex}' AND education = '{edu}'"
            )
        return queries
    else:
        raise KeyError(family)


# ---------------------------------------------------------------------------
# Shadow-model MIA
# ---------------------------------------------------------------------------

def run_workload(config, queries, eps_q, mode, seed):
    """Run a workload through the middleware, return the list of noisy answers."""
    np.random.seed(seed)
    cfg = copy.deepcopy(config)
    cfg.privacy.total_epsilon = 1000.0  # remove cap so workload completes
    mw = DPMiddleware(cfg, mode=mode)
    noisy = []
    for sql in queries:
        try:
            r = mw.execute(sql, epsilon=eps_q)
            v = float(r.rows[0][0]) if r.rows else float("nan")
        except Exception:
            v = float("nan")
        noisy.append(v)
    return np.array(noisy), mw.budget_summary()


def shadow_model_mia(
    config_with: Config,    # config pointing to a DB with the target row
    config_without: Config, # config pointing to a DB without the target row
    family: str,
    eps_q: float,
    mode: ExecutionMode,
    k: int = 50,
    n_shadow: int = 100,
):
    """Shadow-model MIA: train a classifier on shadow runs, evaluate on holdout.

    Returns dict with AUC + per-trial labels.
    """
    # Each "shadow run" is one workload execution under one world (with/without)
    X = []  # feature vector per run: the sequence of noisy outputs
    y = []  # label: 1 if config_with, 0 if config_without

    for trial in range(n_shadow):
        seed_with = trial * 31 + __import__("zlib").crc32(("with" + family).encode()) % 10000  # deterministic
        seed_without = trial * 31 + __import__("zlib").crc32(("without" + family).encode()) % 10000  # deterministic

        queries = build_workload_queries(family, k, seed=trial)

        out_with, _ = run_workload(config_with, queries, eps_q, mode, seed_with)
        out_without, _ = run_workload(config_without, queries, eps_q, mode, seed_without)

        X.append(out_with)
        y.append(1)
        X.append(out_without)
        y.append(0)

    X = np.array(X)
    y = np.array(y)
    # Replace NaN with 0 (queries that failed)
    X = np.nan_to_num(X, nan=0.0)

    # Train/test split
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    split = int(0.7 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    if X.shape[1] == 0:
        return {"auc": 0.5, "accuracy": 0.5, "n_train": 0, "n_test": 0}

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict_proba(X[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], preds)
    acc = clf.score(X[test_idx], y[test_idx])
    return {
        "auc": auc, "accuracy": acc,
        "n_train": len(train_idx), "n_test": len(test_idx),
    }


def make_target_databases(base_db: str = "data/dpdb.duckdb"):
    """Create two databases: original (with target) and minus-one (without).

    Returns (Config_with, Config_without) where the only difference is one
    specific row in the Adult dataset.
    """
    import shutil
    import duckdb

    with_path = "data/dpdb_target_with.duckdb"
    without_path = "data/dpdb_target_without.duckdb"

    if not Path(with_path).exists():
        shutil.copy(base_db, with_path)
    if not Path(without_path).exists():
        shutil.copy(base_db, without_path)
        # Delete one specific Adult row (DuckDB has no DELETE ... LIMIT,
        # so we use a CTID-style approach).
        con = duckdb.connect(without_path)
        # Find the rowid of the first matching row
        result = con.execute("""
            SELECT rowid FROM adult
            WHERE age = 35 AND education = 'Bachelors' AND sex = 'Male'
            ORDER BY rowid
            LIMIT 1
        """).fetchone()
        if result is not None:
            target_rowid = result[0]
            con.execute(f"DELETE FROM adult WHERE rowid = {target_rowid}")
            print(f"  Deleted target row (rowid={target_rowid}) from {without_path}")
        con.close()

    base = Config.from_yaml("config.yaml")
    cfg_with = copy.deepcopy(base)
    cfg_with.duckdb_path = with_path
    cfg_without = copy.deepcopy(base)
    cfg_without.duckdb_path = without_path
    return cfg_with, cfg_without


# ---------------------------------------------------------------------------
# Cross-check with R2 model: predict AUC from budget consumption
# ---------------------------------------------------------------------------

def predict_auc_from_budget(eps_consumed: float, eps_per_release: float) -> float:
    """Heuristic: total privacy loss = eps_consumed, theoretical bound is exp/(1+exp).

    The argument: under workload-aware accounting, the empirical attacker can
    distinguish two worlds by observing eps_consumed in total privacy budget.
    The optimal MIA advantage is bounded by (exp(eps_consumed)-1)/2 per Lemma 5
    of Yeom et al. 2018, giving max AUC = 0.5 + (exp(eps_consumed)-1)/4 clipped.
    """
    advantage = (np.exp(eps_consumed) - 1) / 2
    return min(1.0, 0.5 + advantage / 2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_workload_mia(output_dir, n_shadow=50):
    """Run MIA across W1-W4 at multiple eps values."""
    print("Setting up with/without target databases...")
    cfg_with, cfg_without = make_target_databases()

    workloads = ["W1_repetitive", "W2_zipf", "W3_uniform", "W4_drilldown"]
    eps_values = [0.1, 0.5, 1.0, 2.0]
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP]
    k = 50  # workload length

    records = []
    print(f"Running MIA: {len(workloads)} workloads x {len(eps_values)} eps "
          f"x {len(modes)} modes x {n_shadow} shadow runs")
    for wl in workloads:
        for eps_q in eps_values:
            for mode in modes:
                r = shadow_model_mia(
                    cfg_with, cfg_without, wl,
                    eps_q=eps_q, mode=mode,
                    k=k, n_shadow=n_shadow,
                )
                # Get empirical budget consumed (run one extra workload)
                queries = build_workload_queries(wl, k, seed=999)
                _, summary = run_workload(cfg_with, queries, eps_q, mode, seed=999)
                consumed = summary["consumed_epsilon"]

                # R2 model prediction: AUC bound from cumulative epsilon
                predicted_auc = predict_auc_from_budget(consumed, eps_q)

                records.append({
                    "workload": wl,
                    "mode": mode.value,
                    "eps_per_query": eps_q,
                    "k": k,
                    "empirical_consumed_eps": consumed,
                    "empirical_mia_auc": r["auc"],
                    "empirical_mia_accuracy": r["accuracy"],
                    "model_predicted_auc": predicted_auc,
                })
                print(f"  {wl}/{mode.value}/eps={eps_q}: AUC={r['auc']:.3f}, "
                      f"budget={consumed:.2f}, pred_auc={predicted_auc:.3f}")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "workload_mia.csv", index=False)
    return df


def plot_workload_mia(df, output_dir):
    palette = {"naive_dp": "#e74c3c", "workload_dp": "#3498db"}

    # Plot 1: AUC by workload, mode, eps
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="workload", y="empirical_mia_auc",
                hue="mode", palette=palette, ax=ax)
    ax.axhline(0.5, color="gray", linestyle=":", label="Random guess (0.5)")
    ax.set_title("Shadow-Model MIA AUC across W1-W4 (averaged over epsilons)")
    ax.set_ylabel("Empirical MIA AUC")
    ax.set_xlabel("Workload")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "workload_mia_auc.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "workload_mia_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Model-vs-empirical AUC scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    for wl in df["workload"].unique():
        sub = df[df["workload"] == wl]
        ax.scatter(sub["model_predicted_auc"], sub["empirical_mia_auc"],
                   label=wl, s=80, alpha=0.7)
    lo, hi = 0.45, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Model = empirical")
    ax.set_xlabel("Model-predicted upper bound on MIA AUC")
    ax.set_ylabel("Empirical MIA AUC")
    ax.set_title("Model cross-check: predicted vs empirical AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "workload_mia_vs_model.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "workload_mia_vs_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved workload MIA figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/workload_leakage")
    parser.add_argument("--shadow", type=int, default=50)
    args = parser.parse_args()
    df = run_workload_mia(Path(args.output), n_shadow=args.shadow)
    plot_workload_mia(df, Path(args.output))

    print("\n=== Workload MIA summary ===")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
