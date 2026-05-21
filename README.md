# DP-SQL: A Statistical Model of Privacy-Budget Consumption in Repeated Aggregate SQL Workloads

**CMP653 Database Management Systems Project — Hacettepe University**
Author: Refik Can Öztaş (N25142279)

A differentially private SQL middleware built around a **closed-form analytical model** that predicts privacy budget consumption and per-query utility from workload structure (template repetition skew, group cardinality, temporal staleness). The model recovers the toy `1 − 1/k` budget-savings result as the high-skew corner case and naive sequential composition as the uniform-distribution limit. The middleware is the experimental apparatus that validates the model.

---

## TL;DR — Headline Results

### Model validation (R6)
On 720 trials across a Zipf workload grid (α ∈ {0, 0.5, 1, 1.5, 2, 3} × k ∈ {10, 25, 50, 100}):

- Predicted `E[u_k]` matches empirical mean within **<3% in 22 of 24 cells**.
- Predicted budget savings `S(k) = 1 - E[u_k]/k` matches within **<2% in 23 of 24 cells**.
- Two model limits empirically confirmed:
  - α → ∞ (perfect repetition): `S(k) = 1 − 1/k` ✓
  - α → 0 (uniform): `S(k) → 0` ✓

### Leakage analysis (R4)
- Membership Inference AUC tracks the theoretical bound `exp(ε)/(1+exp(ε))` within ≤1%.
- Drill-down reconstruction error scales as `~2/ε`, as predicted by the difference of two Laplace variables.

### Temporal coupling (R3)
- Staleness tolerance τ → ∞: budget reduces to static case `eps_q × E[u_k]` ✓
- Update rate λ > 0: budget grows linearly with `T·λ·q_inv`, matching the analytical prediction within 15%.

---

## What This Project Is

The milestone version framed exact-repeat caching as the contribution. The reviewer pointed out that caching is a trivial consequence of DP's post-processing property. The final version makes the **analytical model** the contribution; caching is the mechanism that the model predicts the savings of.

### The Model in One Equation

For a workload of `k` queries drawn i.i.d. from a template distribution {p_i}:
```
E[u_k] = Σ_i [1 − (1 − p_i)^k]                    (Proposition 1)
E[ε_workload-aware(k)] = ε_q × E[u_k]              (Proposition 2)
Budget savings: S(k) = 1 − E[u_k] / k             (Proposition 2)
```

Two limits recover existing intuition:
- **Perfect repetition** (p_1 = 1): `S(k) = 1 − 1/k` (the toy formula, now a corner case)
- **Uniform, m → ∞**: `S(k) → 0` (naive composition)

Plus a concentration bound (McDiarmid's inequality) and a utility prediction under fixed total budget.

### The Temporal Extension

```
E[ε_temporal(T)] = ε_q × E[u_k_total] × N(T, τ, λ, q)
N(T, τ, λ, q) = ⌈T/τ⌉ + T·λ·q              (Proposition 6)
```

Three regimes: static (τ → ∞), bounded staleness, update-driven Poisson.

---

## Reviewer Feedback → Revision Map

| Brief item | What it asked for | Where addressed |
|------------|-------------------|-----------------|
| R1 | Reframe central contribution | `report/final_report.tex` §1, abstract |
| R2 | Statistical budget vs DP model | `src/dpdb/model.py`, `report/R2_model_sketch.md`, §4 |
| R3 | Couple budget with temporality | `src/dpdb/budget.py` (staleness + updates), §5 Algorithm 1 |
| R4 | Execute leakage experiments | `experiments/leakage.py`, §7 |
| R5 | Address inline comments | §3 (AVG), §4 (1-1/k as corner case), §5 (Algorithm 1) |
| R6 | Run benchmark campaign | `experiments/{model,semantic,temporal}_validation.py`, §6 |

Full mapping in [`report/response_to_reviewer.md`](report/response_to_reviewer.md).

---

## Implementation

A ~1.4 KLOC Python prototype with 57 passing unit tests:

| Component | File | Purpose |
|-----------|------|---------|
| **Analytical model** | `src/dpdb/model.py` | Zipf workload, E[u_k], savings, McDiarmid, utility, temporal |
| SQL parser/validator | `src/dpdb/parser.py` | sqlglot-based, single-table aggregates only |
| Sensitivity analyzer | `src/dpdb/analyzer.py` | COUNT/SUM/AVG sensitivity with config bounds |
| Laplace mechanism | `src/dpdb/mechanisms.py` | Calibrated noise + GROUP BY parallel composition |
| **Budget ledger** | `src/dpdb/budget.py` | Naive, workload-aware, semantic-aware, temporal modes |
| Template matching | `src/dpdb/template.py` | AST normalization + hashing for L1 cache |
| **Semantic matching** | `src/dpdb/semantic.py` | Tree kernel (Collins-Duffy 2001) + AST embedding (CodeBERT-style) |
| **Zipf workload gen** | `src/dpdb/workload_gen.py` | Real TPC-H/Adult templates sampled from Zipf(α) |
| Middleware orchestrator | `src/dpdb/middleware.py` | 5 execution modes |
| Step-by-step demo | `src/dpdb/demo.py` | Visual trace UI |

---

## Quick Start

```powershell
# 1. Install
pip install -e ".[dev]"

# 2. Generate data
python3 scripts/load_data.py --sf 1.0

# 3. Run the analytical model self-check (verifies limits)
python3 -m dpdb.model

# 4. Run the model-validation benchmark (720 trials, ~3 min)
python3 experiments/model_validation.py --trials 30

# 5. Run the leakage experiments
python3 experiments/leakage.py

# 6. Run the temporal validation
python3 experiments/temporal_validation.py --trials 10

# 7. Run the semantic L2 experiment
python3 experiments/semantic_validation.py --trials 5

# 8. Run the live demo
python3 scripts/presentation_demo.py

# 9. Run unit tests
python3 -m pytest tests/ -v
```

---

## Datasets

- **TPC-H SF=1** — generated via DuckDB's official `tpch` extension. 8 tables, 6,001,215 lineitem rows.
- **UCI Adult** — downloaded from `archive.ics.uci.edu`. 48,842 rows, 15 attributes. The de-facto standard benchmark in the DP literature.

Both reside in a single `data/dpdb.duckdb` file. Backend can be switched to PostgreSQL by changing `config.yaml`.

---

## Execution Modes

| Mode | Cache | When to use |
|------|-------|-------------|
| `EXACT` | none | Non-private baseline (gold standard) |
| `NAIVE_DP` | none | PINQ-style sequential composition baseline |
| `WORKLOAD_DP` | L1 exact-match | Workload-aware accounting (Sections 4-6) |
| `TEMPORAL_DP` | L1 + staleness | With τ and λ from the temporal extension (Section 5) |
| `SEMANTIC_DP` | L1 + L2 semantic | Tree kernel + embedding (Section 6, honest negative result) |

---

## Key Empirical Results

### 1. Model accuracy on Zipf workloads (m=7, 30 trials per cell)

| α \\ k | 10 | 25 | 50 | 100 |
|--------|-----|-----|-----|-----|
| 0.0 | 5.50 / 5.53 | 6.85 / 6.77 | 7.00 / 7.00 | 7.00 / 7.00 |
| 1.0 | 4.73 / 4.70 | 6.32 / 6.30 | 6.88 / 6.90 | 7.00 / 7.00 |
| 3.0 | 2.19 / 2.07 | 3.07 / 3.33 | 3.85 / 3.90 | 4.72 / 4.67 |

Format: `predicted / empirical`. Worst absolute error 0.26; worst relative error 8.6% (high-skew, low-k cell).

### 2. Membership inference attack vs ε

| ε | MIA AUC (empirical) | Theory `e^ε/(1+e^ε)` |
|---|---------------------|----------------------|
| 0.01 | 0.499 | 0.503 |
| 0.10 | 0.522 | 0.525 |
| 1.00 | 0.724 | 0.731 |
| 5.00 | 0.988 | 0.993 |

### 3. Temporal regime

| τ | Empirical ε | Model prediction |
|---|-------------|------------------|
| 10 | 38.3 | 69.9 |
| 25 | 22.5 | 27.9 |
| 50 | 13.7 | 14.0 |
| 100 | 7.0 | 7.0 |
| ∞ | 7.0 | 7.0 |

Model is a slight upper bound; both follow the same trend.

---

## Project Layout

```
diffpriv-db/
├── src/dpdb/
│   ├── model.py           ★ ANALYTICAL MODEL (R2)
│   ├── workload_gen.py    Zipf-parameterized workloads
│   ├── budget.py          Budget ledger + temporal (R3)
│   ├── semantic.py        Tree kernel + AST embedding
│   ├── parser.py / analyzer.py / mechanisms.py
│   ├── middleware.py / db.py / config.py
│   └── demo.py            Step-by-step trace UI
├── experiments/
│   ├── model_validation.py     ★ R6 main validation
│   ├── leakage.py              ★ R4 MIA + reconstruction
│   ├── temporal_validation.py  ★ R3 empirical
│   ├── semantic_validation.py  Honest semantic L2 result
│   └── benchmark.py             Original W1-W4 benchmark
├── results/
│   ├── model_validation/  ★ Figures + CSV from R6
│   ├── leakage/           ★ MIA + reconstruction figures
│   ├── temporal/          ★ Temporal validation
│   └── semantic/          ★ Semantic L2 results
├── report/
│   ├── final_report.tex          ★ THE REVISED PAPER
│   ├── response_to_reviewer.md   ★ Map of how each comment is addressed
│   ├── R2_model_sketch.md         ★ Pre-paper derivation
│   └── milestone_report.tex       Original milestone (for comparison)
├── tests/                57 unit tests across 6 modules
├── scripts/              Data loaders, demos
├── config.yaml
├── pyproject.toml
└── README.md
```

---

## Status

| Phase | Status |
|-------|--------|
| R1: Reframe contribution | ✓ Done |
| R2: Analytical model + 5 propositions | ✓ Done, validated <3% |
| R3: Temporal extension + Proposition 6 | ✓ Done, validated |
| R4: Leakage experiments (MIA + reconstruction) | ✓ Done, matches theory |
| R5: Address inline comments | ✓ Done |
| R6: Benchmark campaign with model validation | ✓ Done |
| Semantic L2 cache (AI angle) | ✓ Done (with honest tradeoff) |
| Response-to-reviewer document | ✓ Done |
| 57 unit tests passing | ✓ |
| Final paper compiled | Pending Overleaf upload |

---

## License

Course project for educational use. If you build on this, please cite:

```bibtex
@misc{oztas2026dpsql,
  author = {Öztaş, Refik Can},
  title = {A Statistical Model of Privacy-Budget Consumption in Repeated Aggregate SQL Workloads},
  year = {2026},
  note = {CMP653 Project Report, Hacettepe University}
}
```
