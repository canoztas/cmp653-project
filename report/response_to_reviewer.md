# Response to Reviewer

**Project:** A Statistical Model of Privacy-Budget Consumption in Repeated Aggregate SQL Workloads
**Course:** CMP653 Database Management Systems, Hacettepe University
**Submission stage:** Final report (revised after milestone review)

This document maps each comment from the annotated milestone PDF to the revision that addresses it.

---

## Inline comments on the milestone draft

### Comment 1 — Abstract, "workload-aware budget ledger with exact-repeat caching": *"what is the use of caching in DP?"*

**Where addressed.** Abstract (rewritten), §1 Introduction (`What this report contributes`), §4 Analytical Model.

**Action.** Caching is no longer the contribution. The contribution is the **analytical model** that predicts when and by how much exact-repeat reuse helps. The model gives a closed-form expression $\E[\eps_{\mathrm{wa}}(k)] = \eps_q \cdot \E[u_k]$ with $\E[u_k] = \sum_i [1-(1-p_i)^k]$, parameterized by workload structure. Caching becomes the mechanism that the model predicts the savings of, not a standalone claim.

---

### Comment 2 — Intro research question: *"not properly explored in text!"*

**Where addressed.** §1 (last paragraph), §7 Empirical Validation (`Research question answered`).

**Action.** The research question is now restated explicitly and answered in §7. The new RQ is: *"Given a workload distribution and a privacy budget, can a closed-form statistical model predict the realized budget consumption and per-query utility, and does it explain when workload-aware accounting beats naive composition?"*

It is answered in §7 with empirical validation showing model–empirical agreement within $<3\%$ across 21 of 24 cells of the $(\alpha, k)$ grid.

---

### Comment 3 — Algorithm 1: *"trivial!"*

**Where addressed.** Algorithm 1 in §6 System Implementation (renamed to capture extra logic).

**Action.** The pseudocode now includes (a) logical-clock advancement, (b) update event simulation with per-entry invalidation, (c) staleness check `age > tau`, (d) a per-aggregate budget split `eps_q/n_agg` with parallel composition across GROUP BY groups (AVG remains a single conservative Laplace release; the `2*eps_q`-per-group decomposition is future work, not implemented — see Comment 5 and §11). It is no longer reducible to "parse, cache, return"—it explicitly couples cache validity with the temporal regime of §5 and the per-aggregate accounting of §3.

---

### Comment 4 — Section 3, `1 − 1/k` derivation: *"?"*

**Where addressed.** §4 Proposition 2 and Limit A.

**Action.** The toy `1 − 1/k` formula is now Limit A of Proposition 2 (eq.~(5) in the report), derived as the corner case when the workload distribution collapses to a single template ($p_1 = 1$). It is no longer a standalone result.

---

### Comment 5 — AVG implemented via noisy SUM / noisy COUNT: *"why?"*

**Where addressed.** §3 Threat Model and Preliminaries, AVG bullet; §11 Future Work, "Decomposed AVG" paragraph.

**Action — and an honest correction.** The milestone paper described AVG as "implemented through bounded noisy SUM and noisy COUNT". The current implementation (`src/dpdb/middleware.py`, line 227) does NOT actually decompose: it applies the Laplace mechanism directly to the AVG value with sensitivity equal to the column upper bound `B_c` (the worst case, group of size one). The decomposed SUM/COUNT version that would amortize noise by the group size `n` is deferred to Future Work and clearly flagged in §11. The §3 discussion now (i) discloses the current conservative mechanism honestly and (ii) justifies why explicit per-group accounting is the right design direction without overstating that it is already implemented. The discrepancy the reviewer would have caught is now visible in the paper itself.

---

### Comment 6 — Section 5, "caching is already working like that!"

**Where addressed.** §1 Introduction (`What this report contributes`), §4 (Limit A note).

**Action.** Caching is now explicitly demoted to a mechanism. The contribution is the model that explains when caching helps. The milestone's central claim "exact-repeat reuse saves budget" is no longer the headline; it is Proposition 2, Limit A.

---

## Summary review (handwritten)

### Comment 7 — *"Assumptions on exact-repeat reuse are very strong. Not realistic."*

**Where addressed.** §5 Temporal Extension (new section).

**Action.** Three temporal regimes are formalized: static snapshot, bounded staleness `tau`, update-driven Poisson `lambda`. The reviewer's pushback now defines an entire section. Algorithm 1 implements all three.

---

### Comment 8 — *"Budget and temporality should be considered together."*

**Where addressed.** §5 Temporal Extension, Proposition 6.

**Action.** Proposition 6 gives a single expression
`E[eps_temporal(T)] = eps_q * E[u_total] * N(T, tau, lambda, q_inv)`
where `N` couples horizon, staleness tolerance, and update rate. Setting `tau = inf` and `lambda = 0` recovers the static-snapshot case.

---

### Comment 9 — *"Leakage and savings models are very limited."*

**Where addressed.** §9 Privacy Leakage Analysis (new section), plus a workload-level subsection added in response to the explicit "across W1-W4 at multiple eps values" request.

**Action.** Three empirical leakage analyses are now executed (not "planned"):

- **Single-query MIA** AUC across `eps in {0.01, ..., 5}` against the theoretical bound `exp(eps)/(1+exp(eps))`. Empirical curve tracks theory within `<=1%`. This is the optimal MIA for a single Laplace release (the likelihood ratio is monotone in the observation, so threshold = optimal classifier; shadow models would not help here).
- **Reconstruction** on a 5-level drill-down, against the theoretical scaling `2/eps`. Empirical curve matches within `<=10%`.
- **Workload-level shadow-model MIA** across W1-W4 at `eps in {0.1, 0.5, 1.0, 2.0}` and both NAIVE_DP and WORKLOAD_DP modes (32 cells x 60 shadow runs each). The shadow model is a logistic-regression classifier trained on simulated runs against two world instances. Empirical AUC ranges from 0.14-0.58, substantially below the cumulative-budget bound (which approaches 1.0). This honest negative finding shows that cumulative-eps bounds are *loose* for realistic workload MIA -- a more interesting result than just confirming the bound.

**Cross-check with R2 model.** §9 now includes a model-vs-empirical AUC comparison: the R2 model gives an upper bound on attacker advantage from cumulative eps; empirical AUC is uniformly below it. This demonstrates that the model is conservative for leakage prediction (errs toward overstating risk, which is the safe side).

---

### Comment 10 — *"Any model on budget vs DP? Propose statistical model."*

**Where addressed.** §4 Analytical Model (entirely new), §7 Empirical Validation.

**Action.** This is the load-bearing change of the revision. Five propositions plus one temporal extension. Model code in `src/dpdb/model.py`; validation in `experiments/model_validation.py`; figures in `results/model_validation/`.

---

## What was preserved (per the approved framing)

- Single-table aggregate SQL subset (no expansion to joins / subqueries).
- "Simpler systems path" over relational joint optimization.
- Database systems framing over PostgreSQL/DuckDB.
- TPC-H as the relational substrate.

---

## Where each revision lives in the artifact

| Brief item | Code/text |
|-----------|-----------|
| R1 (reframe) | `report/final_report.tex` §1, abstract |
| R2 (model) | `src/dpdb/model.py`, `tests/test_model.py`, `report/R2_model_sketch.md`, `final_report.tex` §4 |
| R3 (temporal) | `src/dpdb/budget.py` (staleness + update simulation), `src/dpdb/model.py` (`TemporalRegime`), `final_report.tex` §5 |
| R4 (leakage) | `experiments/leakage.py`, `results/leakage/`, `final_report.tex` §9 |
| R5 (inline) | `final_report.tex` §1, §3, §4, §6, §7 (each addressed in the body) |
| R6 (campaign) | `experiments/model_validation.py`, `results/model_validation/`, `final_report.tex` §7 |
| Reproducibility | this README map + `experiments/*.py` scripts with deterministic seeds |

---

## Outstanding follow-ups (acknowledged honestly)

- Real workloads exhibit burstiness rather than i.i.d. template sampling; the model assumes the simpler case. Burstiness modelling is a tractable extension via a renewal-process replacement for Poisson arrivals.
- The leakage analysis uses simulated counts rather than the live TPC-H data for the differencing attack; the comparison would carry more weight on real workloads. Listed in §8 Discussion.
- The cross-check between the analytical model and the leakage curve is qualitative (drill-down kills caching and admits reconstruction; high-skew enables both better savings and tighter `eps_q`). A formal joint bound is open work.
