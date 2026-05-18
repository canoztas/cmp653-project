# Demo Guide for Project Presentation

## Quick Commands

### Full presentation (4 scenarios)
```powershell
python3 scripts/presentation_demo.py
```
Press Enter to advance through each scenario. Use `--auto` to run without pauses.

### Run a single scenario
```powershell
python3 scripts/presentation_demo.py --scenario 1   # step-by-step trace
python3 scripts/presentation_demo.py --scenario 2   # cache hit demo
python3 scripts/presentation_demo.py --scenario 3   # 3-way baseline comparison
python3 scripts/presentation_demo.py --scenario 4   # 20-query budget savings
```

### Custom interactive demo
```powershell
# Step-by-step trace of any query
python3 -m dpdb.demo --query "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount > 0.05"

# Side-by-side comparison
python3 -m dpdb.demo --compare --query "SELECT COUNT(*) FROM adult WHERE age > 40"

# Interactive REPL with tracing
python3 -m dpdb.demo
```

## What Each Scenario Shows

| # | Scenario | Key Message |
|---|----------|-------------|
| 1 | Single query trace | The 9-step DP pipeline: parse, template, cache check, sensitivity, budget, exec, noise, store |
| 2 | Repeated query | Cache hit returns the same noisy answer at eps=0 (post-processing property) |
| 3 | Baseline comparison | Exact vs PINQ/Naive vs Ours -- 3 repeated queries, ours saves 67% budget |
| 4 | Workload savings | 20 repeated queries: Naive answers 10/20, Ours answers 20/20 (90% savings) |

## Headline Numbers to Mention

* Dataset sizes: TPC-H SF=1 (6M lineitem rows) + UCI Adult (48K rows)
* W1 Repetitive workload: **10x budget reduction** (10.0 -> 1.0 eps)
* W2 Parametric workload: **3.3x budget reduction** with 85% cache hit rate
* All 28 unit tests passing
* Implementation: ~1000 lines Python, single-file DuckDB backend

## Baselines Compared

1. **Exact SQL** -- non-private gold standard
2. **PINQ / Naive DP** -- textbook sequential composition (every query consumes its full eps)
3. **Workload-Aware DP (ours)** -- template caching + post-processing property

PINQ (McSherry, SIGMOD 2009) is the canonical reference for naive composition behavior in DP-SQL systems. Our middleware implements that approach as the "naive_dp" baseline and our contribution shows where workload-aware accounting wins.

## Datasets

* **TPC-H** (real, generated via DuckDB's official `tpch` extension at SF=1)
  - 6,001,215 lineitem rows, 1,500,000 orders, 150,000 customers
* **UCI Adult** (real, downloaded from archive.ics.uci.edu)
  - 48,842 rows, 15 attributes -- standard DP literature benchmark

Generated/loaded via:
```powershell
python3 scripts/load_data.py --sf 1.0
```

## Files Reviewer May Want to See

* `src/dpdb/budget.py` -- the workload-aware ledger (core contribution)
* `src/dpdb/template.py` -- template extraction with hash-based matching
* `src/dpdb/middleware.py` -- main orchestrator
* `report/milestone_report.tex` -- formal write-up
* `results/figures/` -- PNG/PDF plots from benchmark
* `experiments/benchmark.py` -- reproducible experiment runner
