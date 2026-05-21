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

**Where addressed.** §1 (last paragraph), §6 Empirical Validation (`Research question answered`).

**Action.** The research question is now restated explicitly and answered in §6. The new RQ is: *"Given a workload distribution and a privacy budget, can a closed-form statistical model predict the realized budget consumption and per-query utility, and does it explain when workload-aware accounting beats naive composition?"*

It is answered in §6 with empirical validation showing model–empirical agreement within $<3\%$ across 22 of 24 cells of the $(\alpha, k)$ grid.

---

### Comment 3 — Algorithm 1: *"trivial!"*

**Where addressed.** Algorithm 1 in §5 System Implementation (renamed to capture extra logic).

**Action.** The pseudocode now includes (a) logical-clock advancement, (b) update event simulation with per-entry invalidation, (c) staleness check `age > tau`, (d) AVG GROUP BY accounting at `2*eps_q per group`. It is no longer reducible to "parse, cache, return"—it explicitly couples cache validity with the temporal regime of §4 and the per-aggregate accounting of §3.

---

### Comment 4 — Section 3, `1 − 1/k` derivation: *"?"*

**Where addressed.** §4 Proposition 2 and Limit A.

**Action.** The toy `1 − 1/k` formula is now Limit A of Proposition 2 (eq.~(5) in the report), derived as the corner case when the workload distribution collapses to a single template ($p_1 = 1$). It is no longer a standalone result.

---

### Comment 5 — AVG implemented via noisy SUM / noisy COUNT: *"why?"*

**Where addressed.** §3 Threat Model and Preliminaries, last bullet (Sensitivities), with sub-paragraphs *"Why split?"* and *"Why is doubling the right honest accounting?"*

**Action.** Two-paragraph justification covering (i) why the alternative direct-mean mechanism leaks `n` itself, and (ii) why explicit `2*eps_q` accounting beats hiding the SUM/COUNT split inside a stateless library. This turns a flagged design choice into a point in favour of the explicit ledger.

---

### Comment 6 — Section 5, "caching is already working like that!"

**Where addressed.** §1 Introduction (`What this report contributes`), §4 (Limit A note).

**Action.** Caching is now explicitly demoted to a mechanism. The contribution is the model that explains when caching helps. The milestone's central claim "exact-repeat reuse saves budget" is no longer the headline; it is Proposition 2, Limit A.

---

## Summary review (handwritten)

### Comment 7 — *"Assumptions on exact-repeat reuse are very strong. Not realistic."*

**Where addressed.** §4 Temporal Extension (new section).

**Action.** Three temporal regimes are formalized: static snapshot, bounded staleness `tau`, update-driven Poisson `lambda`. The reviewer's pushback now defines an entire section. Algorithm 1 implements all three.

---

### Comment 8 — *"Budget and temporality should be considered together."*

**Where addressed.** §4 Temporal Extension, Proposition 6.

**Action.** Proposition 6 gives a single expression
`E[eps_temporal(T)] = eps_q * E[u_total] * N(T, tau, lambda, q_inv)`
where `N` couples horizon, staleness tolerance, and update rate. Setting `tau = inf` and `lambda = 0` recovers the static-snapshot case.

---

### Comment 9 — *"Leakage and savings models are very limited."*

**Where addressed.** §7 Privacy Leakage Analysis (new section).

**Action.** Two empirical leakage analyses are now executed (not "planned"):

- **Membership inference** AUC across `eps in {0.01, ..., 5}` against the theoretical bound `exp(eps)/(1+exp(eps))`. Empirical curve tracks theory within `<=1%`.
- **Reconstruction** on a 5-level drill-down, against the theoretical scaling `sqrt(2)/eps`. Empirical curve matches within `<=10%`.

---

### Comment 10 — *"Any model on budget vs DP? Propose statistical model."*

**Where addressed.** §4 Analytical Model (entirely new), §6 Empirical Validation.

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
| R4 (leakage) | `experiments/leakage.py`, `results/leakage/`, `final_report.tex` §7 |
| R5 (inline) | `final_report.tex` §1, §3, §4, §5, §6 (each addressed in the body) |
| R6 (campaign) | `experiments/model_validation.py`, `results/model_validation/`, `final_report.tex` §6 |
| Reproducibility | this README map + `experiments/*.py` scripts with deterministic seeds |

---

## Outstanding follow-ups (acknowledged honestly)

- Real workloads exhibit burstiness rather than i.i.d. template sampling; the model assumes the simpler case. Burstiness modelling is a tractable extension via a renewal-process replacement for Poisson arrivals.
- The leakage analysis uses simulated counts rather than the live TPC-H data for the differencing attack; the comparison would carry more weight on real workloads. Listed in §8 Discussion.
- The cross-check between the analytical model and the leakage curve is qualitative (drill-down kills caching and admits reconstruction; high-skew enables both better savings and tighter `eps_q`). A formal joint bound is open work.
