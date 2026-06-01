"""Run benchmarks: 3 modes x 4 workloads, collect metrics."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from experiments.workloads import ALL_WORKLOADS


def run_single_workload(
    config: Config,
    mode: ExecutionMode,
    queries: list[str],
    query_epsilon: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a workload under a given mode and collect per-query metrics."""
    np.random.seed(seed)

    mw = DPMiddleware(config, mode=mode)

    records = []
    for i, sql in enumerate(queries):
        # Get exact result for comparison
        exact_mw = DPMiddleware(config, mode=ExecutionMode.EXACT, db=mw.db)
        exact_result = exact_mw.execute(sql)

        # Run under the target mode
        result = mw.execute(sql, epsilon=query_epsilon)

        # Compute error metrics for each aggregate value
        if exact_result.rows and result.rows and not result.error:
            for row_idx in range(len(exact_result.rows)):
                if row_idx >= len(result.rows):
                    break
                exact_row = exact_result.rows[row_idx]
                noisy_row = result.rows[row_idx]
                for col_idx in range(len(exact_row)):
                    try:
                        exact_val = float(exact_row[col_idx])
                        noisy_val = float(noisy_row[col_idx])
                        abs_error = abs(noisy_val - exact_val)
                        rel_error = abs_error / abs(exact_val) if exact_val != 0 else float("inf")
                    except (TypeError, ValueError):
                        continue

                    records.append({
                        "query_idx": i,
                        "sql": sql[:80],
                        "mode": mode.value,
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "exact_value": exact_val,
                        "noisy_value": noisy_val,
                        "abs_error": abs_error,
                        "rel_error": rel_error,
                        "epsilon_used": result.epsilon_used,
                        "cache_hit": result.cache_hit,
                        "latency_ms": result.latency_ms,
                        "error_msg": result.error or "",
                    })
        elif result.error:
            records.append({
                "query_idx": i,
                "sql": sql[:80],
                "mode": mode.value,
                "row_idx": 0,
                "col_idx": 0,
                "exact_value": None,
                "noisy_value": None,
                "abs_error": None,
                "rel_error": None,
                "epsilon_used": result.epsilon_used,
                "cache_hit": result.cache_hit,
                "latency_ms": result.latency_ms,
                "error_msg": result.error,
            })

    budget_info = mw.budget_summary() or {}
    return pd.DataFrame(records), budget_info


def run_all_experiments(
    config: Config,
    output_dir: Path,
    query_epsilon: float = 1.0,
    n_trials: int = 5,
):
    """Run all workloads x modes x trials and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [ExecutionMode.NAIVE_DP, ExecutionMode.WORKLOAD_DP]

    all_results = []
    all_budgets = []

    for wl_name, wl_func in ALL_WORKLOADS.items():
        queries = wl_func() if callable(wl_func) else wl_func
        print(f"\n{'='*60}")
        print(f"Workload: {wl_name} ({len(queries)} queries)")

        for mode in modes:
            for trial in range(n_trials):
                seed = trial * 1000 + __import__("zlib").crc32(wl_name.encode()) % 1000  # deterministic
                print(f"  Mode: {mode.value}, Trial: {trial+1}/{n_trials}")

                df, budget = run_single_workload(
                    config, mode, queries,
                    query_epsilon=query_epsilon, seed=seed,
                )
                df["workload"] = wl_name
                df["trial"] = trial
                all_results.append(df)

                budget["workload"] = wl_name
                budget["trial"] = trial
                all_budgets.append(budget)

    # Combine and save
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_dir / "results.csv", index=False)

    budgets_df = pd.DataFrame(all_budgets)
    budgets_df.to_csv(output_dir / "budgets.csv", index=False)

    print(f"\nResults saved to {output_dir}")
    return results_df, budgets_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    run_all_experiments(
        config,
        output_dir=Path(args.output),
        query_epsilon=args.epsilon,
        n_trials=args.trials,
    )
