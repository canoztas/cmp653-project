# DP-SQL: A Statistical Model of Privacy-Budget Consumption in Repeated Aggregate SQL Workloads

**CMP653 Database Management Systems Project — Hacettepe University**
Author: Refik Can Öztaş (N25142279)

A Python middleware that intercepts aggregate SQL queries (COUNT, SUM, AVG), adds calibrated Laplace noise for differential privacy, and uses **a closed-form analytical model of budget consumption** to:
1. predict workload cost before deployment,
2. drive a model-based **adaptive budget allocator**,
3. cross-check empirical privacy leakage against theoretical bounds.

---

## The problem

Aggregate SQL queries like `COUNT`, `SUM`, and `AVG` look harmless — they only ever return a single number, never an individual row. But two overlapping aggregates can quietly unmask one person: ask "how many patients are HIV-positive?" and then "how many patients *other than Alice* are HIV-positive?", and the difference between the two answers reveals Alice's status. Differential privacy stops this by adding calibrated noise to every answer, but each noisy answer spends a slice of a finite **privacy budget** — and on a busy dashboard that budget drains fast.

## The idea

The amount of budget a workload spends is a *structural property of the workload itself*, not of the database. If you know how often each query template tends to repeat, you can predict — in closed form, before running a single query — how much budget the workload will consume and how much accuracy you can buy with it. Exact-repeat caching (reuse a noisy answer for free) is not a separate trick here; it is just the extreme case where every query is identical.

## How it works

- **The closed-form model.** Given a distribution `{p_i}` over query templates and a workload of `k` queries, the expected number of *distinct* templates that actually run is `E[u_k] = Σ_i [1 − (1 − p_i)^k]`. Since you only pay budget the first time a template is seen, the expected budget is `E[ε_wa] = ε_q · E[u_k]` (per-query budget × distinct templates), and the savings versus naive accounting is `S(k) = 1 − E[u_k]/k`. At high skew this approaches the `1 − 1/k` caching limit; at uniform skew it collapses to naive composition (you pay for every query).
- **The middleware pipeline.** Each incoming query is parsed and validated against the supported aggregate subset, hashed by (template, parameters), and checked against the cache. A hit is returned for free (post-processing, zero budget cost); a miss is charged `ε_q`, executed, perturbed with Laplace noise, and stored.
- **The predictive allocator.** Instead of fixing the per-query budget up front, the allocator estimates the distinct-template count `Û` at runtime and sets `ε_q = B / Û` (total budget over expected unique queries), spreading the budget to minimize error — and explicitly rejecting late queries once the budget is exhausted rather than silently lying.
- **The temporal extension.** Real data changes, so cached answers go stale. A staleness tolerance `τ` (how long a cached answer stays valid) and a Poisson update rate `λ` (how fast data changes) feed into the same budget expression, so freshness requirements are priced directly into the forecast.

## What we did, step by step

1. Reframed the contribution from "caching saves budget" to "a model that *predicts* budget consumption from the template distribution," with caching as a corollary.
2. Derived the closed-form model — `E[u_k]`, the budget `ε_q · E[u_k]`, and the savings ratio `S(k)` — with proofs and two clean limiting cases.
3. Added a temporal extension that couples budget to data freshness via staleness `τ` and update rate `λ`.
4. Built a unit-tested DP-SQL middleware over DuckDB/PostgreSQL with four execution modes (exact, naive DP, workload-aware DP, temporal DP).
5. Designed a predictive online allocator that sets `ε_q = B/Û` from the model's live estimate.
6. Ran a 4,155-trial validation campaign across Zipf-skewed workloads, four privacy levels, and TPC-H (SF=1 and SF=10) plus the Adult dataset.
7. Stress-tested leakage with single-query and shadow-model membership inference and a five-level differencing/reconstruction attack against theoretical bounds.
8. Tested an AI/AST semantic cache and reported it as an honest negative — structural similarity is not query equivalence.

---

## What This Project Is

The milestone version framed exact-repeat caching as the contribution. The instructor (correctly) flagged that as a trivial consequence of DP's post-processing property. The final version reframes:

> **Caching is a mechanism. The contribution is the *analytical model that predicts when and by how much caching helps*, instantiated in a working middleware over PostgreSQL/DuckDB and TPC-H/Adult.**

Three concrete deliverables sit under that frame:

| # | Deliverable | Lives in | Novelty |
|---|-------------|----------|---------|
| 1 | **Analytical model** linking workload structure → privacy budget | `src/dpdb/model.py` + `report/final_report.tex` §4 | Adapts occupancy/coupon-collector theory to DP-SQL budget; closed-form for arbitrary Zipf $\alpha$ |
| 2 | **Temporal extension** with staleness $\tau$ + update rate $\lambda$ | `src/dpdb/budget.py` (temporal hooks) + paper §5 | First DP-SQL formulation that couples budget with data freshness |
| 3 | **Predictive budget allocator** built on the model | `src/dpdb/predictive.py` + paper §10 | Sets $\eps_q$ at runtime from a forecast of future workload consumption; we frame the forecast as the classic **unseen-species** problem and show a Smoothed Good-Toulmin estimator ~halves the prediction error vs the plug-in (`src/dpdb/predictors.py`) |

Plus three honest experimental findings (the kind of negative results that make a paper credible):

| # | Finding | Why it's interesting |
|---|---------|-----------------------|
| A | **Semantic L2 cache via Tree Kernel + AST Embedding** gives 90% budget savings BUT wrong answers when the matched queries are not alpha-equivalent | First measured trade-off; warns against naive "AI-powered DP cache" designs |
| B | **Workload-level shadow-model MIA AUC** stays at 0.14–0.58, well below the cumulative-$\eps$ bound of 1.0 | Cumulative-budget upper bounds are loose for realistic multi-query attacks; signal-to-noise is governed by per-query $\eps$, not total |
| C | **Model is dataset-scale-independent** (SF=1 vs SF=10 within 0.03 units) and **$\eps_q$-independent** (cancels structurally) | The model captures workload structure, not data scale—useful for capacity planning before deployment |

---

## Headline Numbers

Full benchmark campaign: **4,155 core trials, ~150K queries, six experimental sweeps**.

### Budget consumption (k=100, total ε=100, 30 trials/cell)

| Workload | Naive ε | Workload-aware ε | Savings | Predicted by model |
|----------|---------|-------------------|---------|----------------------|
| W1 Repetitive (1 template) | 100.0 | 1.0 | **100×** | $E[u_k]=1$, $S(k)=1-1/k$ |
| W2 TPC-H returnflag (Zipf α=0.5) | 100.0 | 3.0 | 33× | $E[u_k]\approx 3$ |
| W2 TPC-H priority (Zipf α=1.0) | 100.0 | 5.0 | 20× | $E[u_k]\approx 5$ |
| W2 Adult Zipf α=1.0 | 100.0 | 7.0 | 14.3× | $E[u_k]\approx 7$ |
| W3 Uniform | 100.0 | 7.0 | 14.3× | $E[u_k]= m=7$ (saturated) |
| W4 Drill-down (no repeats) | 100.0 | 100.0 | 1× | $E[u_k]=k$ (model correctly predicts no savings) |

### Model accuracy

- **Main grid (α × k, 30 trials × 24 cells):** model matches empirical mean within `<3%` in **21/24 cells**
- **Cross-scale (SF=1 vs SF=10, 60M lineitem rows):** identical empirical $E[u_k]$ within 0.03 units
- **Epsilon sweep (ε ∈ {0.1, 0.5, 1.0, 2.0}):** model is $\eps_q$-independent, confirmed across 480 trials

### Privacy leakage

- **Single-query MIA AUC** matches theoretical $e^\eps/(1+e^\eps)$ within ≤1%
- **Reconstruction error** (mean) scales as $1.5/\eps$ as predicted
- **Workload-level shadow-model MIA AUC** = 0.14–0.58 (much lower than the cumulative bound 1.0). The single-query attacks track their per-query bounds; the workload-level attack stays near chance, exposing the looseness of the cumulative bound (an honest negative).

### Allocation policy (Zipf, m=20, k=100, B=10, 60 trials)

- **Safe closed-form `ε_q = B/m`**: answers **100%** of queries at fresh-release MAE **2.0**, vs an ε-greedy budget bandit's **3.8** at the same 100%-answered rate (~1.9×, ≈18× SEM, adversarially verified across 108 configs). It reaches the bandit's best-case operating point at **zero exploration cost**; only a `u_k` forecast (oracle, MAE 1.6) safely goes lower.
- The **predictive allocator** (`ε_q = B/Û`) lowers per-query MAE by up to ~17% at low skew, but the gain is statistically significant only for α ≤ 0.5 (paired t-test) — reported honestly as a low-skew result.

---

## Project Components

The repository is a complete artifact: every figure, table, and number in the paper has a script that produces it.

```
src/dpdb/
  model.py          R2 analytical model: E[u_k], savings ratio, temporal extension
  predictive.py     Online predictive budget allocator built on the model
  predictors.py     Alternative u_k estimators (plug-in, Good-Toulmin, Smoothed-GT, Chao1)
  semantic.py       AI/AST layer: Tree Kernel + sentence-transformer embedding
  budget.py         Privacy budget ledger (4 strategies, temporal hooks)
  middleware.py     Orchestrator with 6 execution modes
  parser.py         sqlglot-based SQL parser/validator
  analyzer.py       Sensitivity analysis (COUNT=1, SUM/AVG=bound)
  mechanisms.py     Laplace mechanism
  template.py       Exact-match template extraction & hashing
  db.py             DuckDB + PostgreSQL backends
  demo.py           Step-by-step REPL trace
  cli.py            Interactive query interface

experiments/
  model_validation.py        Main alpha-grid validation (720 trials)
  extended_sweeps.py         Alpha up to 10, eps sweep, k up to 500
  sf10_validation.py         SF=1 vs SF=10 cross-scale validation
  full_campaign.py           6 workloads × 4 eps × 3 modes × 30 trials = 2160
  leakage.py                 Single-query MIA + reconstruction
  workload_leakage.py        Shadow-model MIA across W1–W4 (R4 closure)
  temporal_validation.py     Tau-sweep + lambda-sweep, 30 trials
  semantic_validation.py     L1 vs L2 semantic cache (honest negative)
  predictive_comparison.py   Predictive vs fixed vs naive (new mechanism)
  aggregate_all_results.py   Combine all CSVs → results/REPORT.md

report/
  final_report.tex            12-section paper, 6 embedded figures, 43 references
  response_to_reviewer.md     Comment-by-comment mapping of instructor feedback
  R2_model_sketch.md          Math derivation + limit checks for the model
  milestone_report.tex        Original milestone (kept for reference)

tests/  (86 unit tests, all passing)
  test_parser.py    test_mechanisms.py  test_budget.py    test_template.py
  test_semantic.py  test_model.py       test_predictive.py
```

---

## Six Execution Modes

| Mode | Cache | Adaptive ε? | Use case |
|------|-------|-------------|----------|
| `EXACT` | none | n/a | Non-private gold standard |
| `NAIVE_DP` | none | no | PINQ-style textbook baseline |
| `WORKLOAD_DP` | exact match (L1) | no | Repetitive dashboards, exact reuse |
| `TEMPORAL_DP` | L1 + staleness/update hooks | no | Streaming data, bounded freshness |
| `SEMANTIC_DP` | L1 + L2 semantic (Tree Kernel + AST Embedding) | no | Approximate answers OK |
| `PREDICTIVE_DP` | L1 | **yes** (model-driven) | Unknown workload, want to use full budget |

---

## AI / NLP Components (Section 8 of the Paper)

A semantic L2 cache layer was added on top of the exact-match L1 cache, combining two complementary similarity tools:

- **Tree Kernel** (Collins & Duffy 2001): counts shared subtrees between two SQL ASTs. Deterministic, no training required, captures structural equivalence.
- **AST Embedding** via `sentence-transformers` (`all-MiniLM-L6-v2`): produces a 384-dim dense representation of the canonicalized AST. Captures higher-level semantic similarity, in the spirit of CodeBERT and GraphCodeBERT.

A query is accepted as a semantic cache match only if both scores cross conservative thresholds ($K_{\mathrm{norm}} \geq 0.95$ and cosine $\geq 0.98$).

**Honest finding (reported in the paper):** semantic matching is dangerous as a free DP cache. Two queries can be structurally near-identical (same template, different literals) and still have very different true answers. The semantic L2 cache reused 80% of budget but returned wrong values by thousands of count units. Future work: pair similarity with a symbolic equivalence prover before admitting a match.

---

## Reproducibility: Figure-to-Script Mapping

| Paper artifact | Script | Output |
|----------------|--------|--------|
| Table 1 (Limit A/B verification) | `python3 -m dpdb.model` | console |
| Table 2 (main grid validation) | `experiments/model_validation.py` | `results/model_validation/*.csv,*.pdf` |
| Extended alpha sweep table | `experiments/extended_sweeps.py` | `results/extended/extended_alpha.{csv,pdf,png}` |
| Epsilon sweep table | `experiments/extended_sweeps.py` | `results/extended/epsilon_sweep.csv` |
| Large-k saturation figure | `experiments/extended_sweeps.py` | `results/extended/extended_large_k.{pdf,png}` |
| Cross-scale (SF=1 vs SF=10) | `experiments/sf10_validation.py` | `results/sf10/sf10_validation.{csv,pdf,png}` |
| Full benchmark grid | `experiments/full_campaign.py` | `results/full_campaign/full_campaign_SF1.csv` + 4 figs |
| Temporal regime | `experiments/temporal_validation.py` | `results/temporal/temporal_validation.{csv,pdf,png}` |
| Single-query MIA AUC | `experiments/leakage.py` | `results/leakage/mia_*.{csv,pdf,png}` |
| Reconstruction error | `experiments/leakage.py` | `results/leakage/reconstruction_*.{csv,pdf,png}` |
| Shadow-model MIA across W1–W4 | `experiments/workload_leakage.py` | `results/workload_leakage/*.{csv,pdf,png}` |
| Semantic L2 cache table | `experiments/semantic_validation.py` | `results/semantic/semantic_validation.{csv,pdf,png}` |
| Predictive allocator table | `experiments/predictive_comparison.py` | `results/predictive/predictive_comparison.{csv,pdf,png}` |
| Algorithm 1 (middleware pipeline) | `src/dpdb/middleware.py` + `src/dpdb/budget.py` | source code |
| Propositions 1–6 | `src/dpdb/model.py` | source + `tests/test_model.py` |
| Aggregate report (all numbers) | `experiments/aggregate_all_results.py` | `results/REPORT.md` + `results/ALL_RESULTS.csv` |

---

## Quick Start

```powershell
# 1. Install
pip install -e ".[dev]"

# 2. Generate data
python3 scripts/load_data.py --sf 1.0
python3 scripts/load_sf10.py            # optional, 75 sec, ~2.5 GB

# 3. Verify the model self-check
python3 -m dpdb.model

# 4. Run experiments (each takes 1–10 minutes; safe to skip individual ones)
python3 experiments/model_validation.py     --trials 30
python3 experiments/extended_sweeps.py      --trials 30
python3 experiments/sf10_validation.py      --trials 30
python3 experiments/full_campaign.py        --trials 30 --k 100 --total-eps 100
python3 experiments/leakage.py
python3 experiments/workload_leakage.py     --shadow 30
python3 experiments/temporal_validation.py  --trials 30
python3 experiments/semantic_validation.py  --trials 30
python3 experiments/predictive_comparison.py --trials 30

# 5. Aggregate everything into one report
python3 experiments/aggregate_all_results.py
# → results/REPORT.md, results/ALL_RESULTS.csv (5613 rows)

# 6. Interactive demo (for showing the system live)
python3 scripts/presentation_demo.py

# 7. Run unit tests
python3 -m pytest tests/ -v
# 86 passed
```

---

## Brief Compliance (Instructor's Revision Plan)

Each item from the instructor's revision brief is addressed and traceable.

| Item | Brief request | Where in code/paper |
|------|---------------|---------------------|
| R1 | Reframe contribution from mechanism to model | Abstract, §1, R2_model_sketch.md |
| R2 | Statistical model of budget vs DP (load-bearing) | §4, src/dpdb/model.py, tests/test_model.py |
| R3 | Couple budget with temporality | §5, src/dpdb/budget.py temporal hooks |
| R4 | Leakage analysis with MIA + reconstruction | §7, experiments/leakage.py + workload_leakage.py |
| R5 | Address inline comments (AVG, 1-1/k, Algorithm 1, RQ) | §3 (AVG paragraph), §4 (Limit A for 1-1/k), §6 (new Algorithm 1) |
| R6 | Benchmark grid (W1–W4, SF=1/SF=10, ε sweep, 30 trials) | §6 + experiments/full_campaign.py (4155 core trials total) |
| Deliverable 1 | Revised paper | report/final_report.tex |
| Deliverable 2 | Reproducibility artifact + figure-to-script map | this README's table above |
| Deliverable 3 | Response-to-reviewer | report/response_to_reviewer.md |
| Bonus | Predictive allocator (Section 10) | src/dpdb/predictive.py + experiments/predictive_comparison.py |
| Bonus | Eight items in Section 11 Future Work | report/final_report.tex §11 |

---

## Paper Section Map

The paper (`report/final_report.tex`) follows a clean 12-section narrative:

```
§1  Introduction              ─ why the milestone framing was thin, what the new contribution is
§2  Background & Related     ─ DP foundations + private SQL systems + DP caching + tree kernels
§3  Threat Model              ─ adversary, neighbouring DBs, supported SQL, sensitivities (AVG justified)
§4  Analytical Model          ─ five propositions + limit verification (R2)
§5  Temporal Extension        ─ Proposition 6, three regimes (R3)
§6  System Implementation     ─ middleware architecture + new Algorithm 1 (R5)
§7  Empirical Validation      ─ 4155-trial core campaign + cross-scale (R6)
§8  Semantic Cache             ─ Tree Kernel + AST Embedding + honest negative
§9  Privacy Leakage Analysis  ─ MIA + reconstruction + shadow-model MIA (R4)
§10 Predictive Allocator      ─ model-driven adaptive ε mechanism (new contribution)
§11 Future Work                ─ eight concrete next steps
§12 Discussion & Conclusion    ─ where the model wins, where it loses
```

---

## Status

- **86 unit tests passing**
- **~8,000 experimental trials** across 13 scripts (4,155 in the core six-sweep §7 campaign, ~3,950 in the follow-up §8–§10 experiments). The aggregated long-form CSV at `results/ALL_RESULTS.csv` records 5,613 result rows.
- **24 figures** generated and tracked in `results/`
- **Paper** ready to compile on Overleaf (LaTeX source + embedded figure paths)
- **GitHub:** [canoztas/cmp653-project](https://github.com/canoztas/cmp653-project) (private)

For one-page Turkish summary, see [ProjeOzeti.md](ProjeOzeti.md).
For demo commands during presentation, see [DEMO.md](DEMO.md).
