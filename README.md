# DP-SQL: A Differentially Private SQL Middleware for Repeated Aggregate Queries

**CMP653 Database Management Systems Project — Hacettepe University**
Author: Refik Can Öztaş (N25142279)

A lightweight Python middleware that intercepts aggregate SQL queries, adds calibrated Laplace noise for differential privacy, and uses **workload-aware budget accounting with template-based caching** to outperform standard sequential composition (PINQ-style) by up to **10x in budget efficiency** on repetitive analytical workloads.

---

## TL;DR — Headline Result

Running 20 identical COUNT queries on TPC-H (SF=1, 6M rows) with a budget of ε = 10:

| Strategy | Queries answered | ε consumed | Cache hits |
|----------|------------------|------------|------------|
| PINQ / Naive DP (textbook sequential composition) | **10 / 20** | 10.0 | 0 |
| Workload-Aware (ours) | **20 / 20** | **1.0** | **19** |

**90% privacy budget savings, 2× more queries answered**, while preserving ε-differential privacy through the post-processing property.

---

## Why This Project Matters

Aggregate SQL queries — even when returning only summary statistics — leak individual-level information through three well-known attack vectors:

1. **Differencing Attack.** Two COUNT queries differing by one record isolate that record's attribute.
   ```sql
   Q1: SELECT COUNT(*) FROM patients WHERE disease='HIV'           -- 47
   Q2: SELECT COUNT(*) FROM patients WHERE disease='HIV' AND name<>'Alice'  -- 46
   ⇒ Alice has HIV
   ```
2. **Reconstruction Attack (Dinur–Nissim 2003).** O(n) aggregate queries with insufficient noise can reconstruct the entire database — the *Fundamental Law of Information Recovery*.
3. **Budget Exhaustion in Repeated Workloads.** A dashboard issuing 100 daily queries at ε = 0.1 each burns through ε = 10 total budget in one day under naive sequential composition.

Differential Privacy (DP) addresses these with calibrated noise — but a *practical* DP-SQL system must also manage the privacy budget across a *workload*, not just a single query. That's what this project investigates.

---

## What's Implemented

A complete prototype (~1,200 lines of Python) including:

| Component | File | Purpose |
|-----------|------|---------|
| SQL Parser & Validator | `src/dpdb/parser.py` | sqlglot-based AST extraction, supports COUNT/SUM/AVG + GROUP BY + WHERE |
| Sensitivity Analyzer | `src/dpdb/analyzer.py` | Global sensitivity per aggregate under tuple-level DP |
| Laplace Mechanism | `src/dpdb/mechanisms.py` | Calibrated noise sampling |
| **Budget Ledger** | `src/dpdb/budget.py` | Naive, workload-aware, and semantic-aware strategies |
| **Template Matching** | `src/dpdb/template.py` | AST normalization + hashing for L1 cache lookup |
| **Semantic Matching** | `src/dpdb/semantic.py` | L2 cache via Tree Kernel (Collins-Duffy) + AST Embedding (sentence-transformers) |
| Middleware Orchestrator | `src/dpdb/middleware.py` | End-to-end query lifecycle |
| Interactive CLI | `src/dpdb/cli.py` | REPL for testing |
| **Step-by-step demo** | `src/dpdb/demo.py` | Visual trace of every query stage |
| **Presentation script** | `scripts/presentation_demo.py` | 5 scenario walk-through |
| Data Loader | `scripts/load_data.py` | TPC-H (via DuckDB extension) + UCI Adult |
| Benchmarks | `experiments/benchmark.py` | 4 workloads × 3 modes × 5 trials |
| Visualizations | `experiments/visualize.py` | 6 publication-quality plots |
| Unit Tests | `tests/` | 28 tests, all passing |

---

## Quick Start

### 1. Install
```powershell
pip install -e ".[dev]"
```

### 2. Generate data (TPC-H SF=1 + UCI Adult)
```powershell
python3 scripts/load_data.py --sf 1.0
```
Output: `data/dpdb.duckdb` containing 8 TPC-H tables (6M+ lineitem rows) and the 48,842-row Adult dataset.

### 3. Run the live demo
```powershell
python3 scripts/presentation_demo.py
```
Walks through 5 scenarios with step-by-step traces.

### 4. Run benchmarks
```powershell
python3 experiments/benchmark.py --trials 5
python3 experiments/visualize.py
```
Generates 6 plots in `results/figures/`.

### 5. Try queries interactively
```powershell
python3 -m dpdb.demo
dp-sql> SELECT COUNT(*) FROM adult WHERE age > 40
dp-sql> SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount > 0.05
dp-sql> \budget
```

---

## Three Execution Modes

| Mode | Cache | When to use |
|------|-------|-------------|
| `NAIVE_DP` | none | PINQ-style textbook baseline |
| `WORKLOAD_DP` | L1 exact-match (template hash + param hash) | Repetitive workloads, exact answers |
| `SEMANTIC_DP` | L1 + **L2 semantic** (Tree Kernel + AST Embedding) | Parametric/exploratory workloads, approximate answers OK |

L2 catches structurally similar queries (different literals, reordered predicates) and returns the nearest cached noisy answer at ε=0. Tradeoff: more budget savings, less per-query accuracy. We measure both empirically.

## How the Middleware Works (9 steps per query)

```
Analyst → [1. SQL Parser/Validator]
       → [2. Template Extraction (AST normalization + hashing)]
       → [3a. Budget Ledger: L1 Exact-Match Cache Lookup]
       → [3b. Optional: L2 Semantic Cache (Tree Kernel + AST Embedding)]
            ├── cache HIT  → return cached noisy result (ε=0, post-processing)
            └── cache MISS:
              → [4. Sensitivity Analyzer]
              → [5. Budget Allocation]
              → [6. Execute on DuckDB]
              → [7. Laplace Noise Injection]
              → [8. Cache Store]
              → [9. Return noisy result]
```

### The Core Idea — Why Caching Is Privacy-Safe

By the **post-processing property of DP**: any deterministic or randomized function applied to the output of an ε-DP mechanism, *without additional access to the private data*, also satisfies ε-DP. Since a cached noisy result was produced by the Laplace mechanism, returning it again without re-querying the database costs **ε = 0** in additional privacy budget.

This is the theoretical justification for our workload-aware ledger.

---

## Empirical Results

### Visible-Noise Comparison (Adult, age ≥ 90, true count = 55, ε = 0.1)

| Run | True | Naive DP | Workload-Aware |
|-----|------|----------|----------------|
| #1 | 55 | 52 | **52** (cache miss) |
| #2 | 55 | 78 | **52** (cache hit, ε=0) |
| #3 | 55 | 61 | **52** (cache hit, ε=0) |
| #4 | 55 | 57 | **52** (cache hit, ε=0) |
| #5 | 55 | 43 | **52** (cache hit, ε=0) |
| **Total ε** | 0 | **0.50** | **0.10** |

- Naive DP gives 5 *different* noisy answers (fresh noise each time)
- Workload-Aware gives 1 noisy answer, reused (post-processing)
- **80% budget savings** with the same privacy guarantee

### Workload-Level Results (TPC-H SF=1, 5 trials, ε = 1.0/query, total = 10.0)

| Workload | Naive ε | Workload-Aware ε | Cache Hit Rate | Savings |
|----------|---------|-------------------|----------------|---------|
| W1 Repetitive | 10.0 | **1.0** | 95% | **10×** |
| W2 Parametric | 10.0 | **3.0** | 85% | **3.3×** |
| W3 Diverse | 10.0 | 10.0 | 0% | 1× (expected) |
| W4 Progressive | 10.0 | 10.0 | 44% | — |

### Semantic Matching (L1+L2): Parametric Workload, 5 different age thresholds

| Strategy | Cache hits | ε consumed | Mean answer error |
|----------|------------|------------|-------------------|
| Workload-Aware (L1 only) | 0 / 5 | 5.0 | 0 (exact) |
| **Semantic (L1+L2)** | **4 / 5** (4 semantic) | **1.0** | ~19,544 (approximate) |

Semantic L2 trades per-query accuracy for additional budget efficiency. Useful when dashboards drill through parametric variations of the same template and approximate answers are acceptable.

---

## Datasets

- **TPC-H SF=1** — generated via DuckDB's official `tpch` extension. 8 tables: region, nation, supplier, customer, part, partsupp, orders, lineitem (6,001,215 rows).
- **UCI Adult** — downloaded from `archive.ics.uci.edu`. 48,842 rows, 15 attributes. The de-facto standard benchmark in the DP literature.

---

## Baselines

| Baseline | Source | Role |
|----------|--------|------|
| Exact SQL | direct DuckDB passthrough | Gold standard (truth) |
| **PINQ / Naive DP** | McSherry, SIGMOD 2009 | Textbook sequential composition |
| **Workload-Aware DP** | This project | Template caching + post-processing |

Additional reference systems available in `baselines/` for future expansion:
- **Chorus** (Uber/UVM, SIGMOD 2020) — DP query rewriter
- **Uber Flex** — SQL Differential Privacy
- **OpenDP** (Harvard) — General DP library
- **IBM diffprivlib** — DP primitives

---

## Mathematical Foundation

For a query function `f: D → ℝ` with global sensitivity `Δf`, the **Laplace mechanism** outputs:
```
M(D) = f(D) + Z,   Z ~ Lap(Δf / ε)
```
which satisfies ε-differential privacy.

**Sensitivities used:**
- COUNT: Δf = 1
- SUM(c): Δf = B_c (configured upper bound on |c|, from TPC-H spec)
- AVG(c): decomposed as noisy_SUM / noisy_COUNT

**Composition:**
- Sequential: k queries each with ε_i give total ε = Σ ε_i (naive baseline)
- Parallel: queries over disjoint subsets give total ε = max ε_i (we use this for GROUP BY)
- Post-processing: cached results cost ε = 0 (workload-aware advantage)

---

## Project Structure

```
diffpriv-db/
├── src/dpdb/              # Core middleware
│   ├── parser.py
│   ├── analyzer.py
│   ├── mechanisms.py
│   ├── template.py
│   ├── budget.py
│   ├── middleware.py
│   ├── demo.py           # Step-by-step trace UI
│   ├── cli.py
│   ├── db.py             # DuckDB + PostgreSQL backends
│   └── config.py
├── experiments/
│   ├── workloads.py      # W1-W4 query workloads
│   ├── benchmark.py      # Run all modes x workloads
│   └── visualize.py      # Generate plots
├── scripts/
│   ├── load_data.py      # TPC-H + Adult loader
│   ├── presentation_demo.py  # 5-scenario walkthrough
│   ├── setup_tpch.sql    # PostgreSQL DDL (optional)
│   └── generate_tpch_data.py # synthetic generator (optional)
├── tests/                # 28 unit tests
├── report/
│   └── milestone_report.tex
├── data/                  # generated, gitignored
├── results/               # generated, gitignored
├── baselines/             # cloned reference systems, gitignored
├── config.yaml
├── pyproject.toml
├── DEMO.md                # demo command cheat-sheet
└── README.md              # this file
```

---

## Evaluation Metrics

Defined across four categories (see [report/milestone_report.tex](report/milestone_report.tex) §6 for formulas):

**Utility**
- MAE (Mean Absolute Error), RMSE, Mean Relative Error
- Answer Quality Rate: fraction of answers within τ relative error

**Budget Efficiency**
- Queries per Epsilon (QPE)
- Budget Exhaustion Point (BEP) — query index where budget runs out
- Cache Hit Rate (CHR)
- Budget Savings Ratio (BSR) — fraction saved vs naive

**Privacy Leakage**
- Membership Inference Accuracy
- Empirical Privacy Loss (log-likelihood ratio)
- Reconstruction Accuracy under simulated attack

**System Performance**
- Latency (median, p95)
- Middleware overhead vs exact SQL
- Throughput

---

## Key References

1. C. Dwork, F. McSherry, K. Nissim, A. Smith. *Calibrating Noise to Sensitivity in Private Data Analysis*. TCC 2006.
2. I. Dinur, K. Nissim. *Revealing Information while Preserving Privacy*. PODS 2003.
3. F. McSherry. *Privacy Integrated Queries: An Extensible Platform for Privacy-preserving Data Analysis*. SIGMOD 2009.
4. C. Dwork, A. Roth. *The Algorithmic Foundations of Differential Privacy*. F&T 2014.
5. W. Dong, D. Sun, K. Yi. *Better than Composition: How to Answer Multiple Relational Queries under DP*. SIGMOD 2023.
6. J. Yu et al. *DOP-SQL: A General-purpose, High-utility, and Extensible Private SQL System*. VLDB 2024.
7. NIST SP 800-226. *Guidelines for Evaluating Differential Privacy Guarantees*. 2025.

Full list (18 references) in [report/milestone_report.tex](report/milestone_report.tex).

---

## Status

| Phase | Status |
|-------|--------|
| SQL parser, sensitivity, Laplace mechanism | ✓ Complete |
| Budget ledger (naive + workload-aware) | ✓ Complete |
| Template extraction & matching | ✓ Complete |
| Middleware orchestrator + CLI | ✓ Complete |
| 28 unit tests | ✓ All passing |
| TPC-H SF=1 + Adult dataset loaded | ✓ Complete |
| 4-workload benchmarks (5 trials each) | ✓ Complete |
| Visualization pipeline | ✓ Complete |
| Step-by-step demo + presentation script | ✓ Complete |
| Milestone report (LaTeX, ACM format) | ✓ Submitted |
| Privacy leakage experiments (MIA, reconstruction) | ◯ Planned |
| Rényi DP accounting | ◯ Optional |
| Join-aware heuristic | ◯ Optional |
| Final paper | ◯ Pending |

---

## License & Citation

Course project for educational purposes. If you build on this, please cite:

```
Öztaş, R.C. (2026). A Differentially Private SQL Middleware for
Repeated Aggregate Queries. CMP653 Project Report, Hacettepe University.
```
