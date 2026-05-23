"""Per-stage latency breakdown of the middleware pipeline (item 3).

Instruments each pipeline stage and reports its share of total query time:
  parse, template-extract, sensitivity, cache-lookup, allocate, db-execute,
  noise-add, cache-store.

Goal: identify where the middleware overhead concentrates so future work
can target the right stage.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

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

from dpdb.analyzer import analyze_sensitivity
from dpdb.budget import AllocationStrategy, BudgetLedger
from dpdb.config import Config
from dpdb.db import create_database
from dpdb.mechanisms import laplace_mechanism
from dpdb.parser import parse_query
from dpdb.template import extract_template, template_hash
from dpdb.workload_gen import ADULT_AGE_BAND_TEMPLATES, generate_zipf_workload

sns.set_theme(style="whitegrid", font_scale=1.1)


def time_pipeline(db, ledger, sql, eps_q):
    """Run one query through the pipeline and record per-stage timings."""
    timings = {}
    t0 = time.perf_counter()
    parsed = parse_query(sql)
    timings["parse"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    template_hash(extract_template(parsed))  # canonicalize + hash
    timings["template_extract"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    cached = ledger.try_cache(parsed)
    timings["cache_lookup"] = (time.perf_counter() - t) * 1000

    if cached is not None:
        timings["sensitivity"] = 0
        timings["allocate"] = 0
        timings["db_execute"] = 0
        timings["noise_add"] = 0
        timings["cache_store"] = 0
        timings["total"] = sum(timings.values())
        timings["cache_hit"] = True
        return timings

    t = time.perf_counter()
    sens = analyze_sensitivity(parsed, ledger.config) if hasattr(ledger, "config") else None
    timings["sensitivity"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    try:
        ledger.allocate(parsed, eps_q)
    except Exception:
        pass
    timings["allocate"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    columns, rows = db.execute_with_columns(sql)
    timings["db_execute"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    for r in rows:
        for col_idx in range(len(r)):
            try:
                val = float(r[col_idx])
                laplace_mechanism(val, 1.0, eps_q)
            except Exception:
                continue
    timings["noise_add"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ledger.store_result(parsed, columns, rows, eps_q)
    timings["cache_store"] = (time.perf_counter() - t) * 1000

    timings["total"] = sum(timings.values())
    timings["cache_hit"] = False
    return timings


def run_breakdown(config, output_dir, n_trials=10):
    """Time the pipeline on each workload family."""
    alpha = 1.0
    k = 50
    workloads = {
        "Repetitive (alpha=inf)": [ADULT_AGE_BAND_TEMPLATES[1].instantiate()] * k,
        "Zipf alpha=1": generate_zipf_workload(
            ADULT_AGE_BAND_TEMPLATES, alpha=alpha, k=k, seed=42)[0],
        "Uniform": generate_zipf_workload(
            ADULT_AGE_BAND_TEMPLATES, alpha=0.0, k=k, seed=42)[0],
    }

    records = []
    print("[Latency] Instrumenting middleware stages")
    for wl_name, queries in workloads.items():
        for trial in range(n_trials):
            db = create_database(config)
            db.connect()
            cfg = copy.deepcopy(config)
            cfg.privacy.total_epsilon = 1000.0
            ledger = BudgetLedger(1000.0, AllocationStrategy.WORKLOAD_AWARE)
            ledger.config = cfg  # for sensitivity analyzer
            for sql in queries:
                t = time_pipeline(db, ledger, sql, eps_q=1.0)
                t["workload"] = wl_name
                t["trial"] = trial
                records.append(t)
            db.close()
        print(f"  {wl_name}: done")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "latency_breakdown.csv", index=False)
    return df


def plot_breakdown(df, output_dir):
    stages = ["parse", "template_extract", "cache_lookup", "sensitivity",
              "allocate", "db_execute", "noise_add", "cache_store"]
    # Mean stage time on cache MISSES only (where all stages run)
    miss = df[~df["cache_hit"]]
    if len(miss) > 0:
        avg_by_stage = miss[stages].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_by_stage.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Mean latency per cache-miss query (ms)")
        ax.set_title("Pipeline-stage breakdown (cache MISS path)")
        plt.tight_layout()
        plt.savefig(output_dir / "latency_breakdown_miss.pdf", dpi=150)
        plt.savefig(output_dir / "latency_breakdown_miss.png", dpi=150)
        plt.close()

    # Stacked breakdown per workload
    g = df.groupby("workload")[stages].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    g.plot(x="workload", kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Stage-by-stage latency contribution per workload")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_breakdown_per_workload.pdf", dpi=150)
    plt.savefig(output_dir / "latency_breakdown_per_workload.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/latency")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    df = run_breakdown(config, Path(args.output), args.trials)
    plot_breakdown(df, Path(args.output))

    stages = ["parse", "template_extract", "cache_lookup", "sensitivity",
              "allocate", "db_execute", "noise_add", "cache_store"]

    print("\n=== Latency summary (cache-miss path, mean ms per query) ===")
    miss = df[~df["cache_hit"]]
    print(miss[stages].mean().round(3).to_string())

    print("\n=== Cache-hit path (mean ms per query) ===")
    hit = df[df["cache_hit"]]
    print(hit[stages].mean().round(3).to_string())


if __name__ == "__main__":
    main()
