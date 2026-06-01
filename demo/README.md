# Live DP-SQL Pipeline Demo

An interactive web demo that runs each query through the **real** `src/dpdb`
middleware components over the **real** Adult table and animates it step by step,
so you can watch the privacy budget deplete, cache hits go free, and the noise
get added — live.

![pipeline](https://img.shields.io/badge/steps-Parse%E2%86%92Template%E2%86%92Cache%E2%86%92Sensitivity%E2%86%92%CE%B5%E2%86%92Noise%E2%86%92Cache-blue)

## Run

```bash
pip install flask duckdb          # numpy/sqlglot already come with the project
python demo/app.py                # -> http://127.0.0.1:5000
```

Open the URL in a browser.

Optional headless smoke test of the run flow (catches JS regressions):
`npm install jsdom && node demo/test_ui.js` (with the server running).

## The interface

- **Plain-language explanations** under every step — what it does and why it
  matters — alongside the real technical detail, in **English or Turkish**
  (EN/TR toggle, top-right; remembered across visits).
- **Terminology glossary** — a slide-in drawer (the *Terms / Sözlük* button, the
  `g` key, or any clickable **term chip**) with searchable, grouped definitions
  (privacy primitives, workload model, caching/temporal, predicting the budget),
  bilingual, mirroring the paper's Appendix A.
- **Colour semantics** everywhere (badges, step icons, chart, timeline): blue =
  spend, green = free cache hit, amber = miss, purple = L2 semantic, red =
  reject/error. A connector rail fills as the query progresses.
- The explanatory content lives in `demo/content.json` (verified against the
  code); the live technical detail, SQL, hashes and numbers always come from the
  real backend.

## What you see

The page animates the query through **exactly the steps `DPMiddleware.execute()`
takes** — it drives the real ledger, predictor, semantic matcher, DB and
`_add_noise`, so it *is* the real system, narrated. The steps shown depend on the
mode:

0. **Mode** — which of the six execution modes is active.
1. **Parse & validate** — COUNT/SUM/AVG + WHERE + GROUP BY; each aggregate's
   SELECT position is recorded (HAVING / COUNT(DISTINCT) are rejected here).
2. **Predictive allocate (pre-cache)** *(predictive)* — `ε_q = B/Û` from the
   occupancy model, before the cache, with the live `Û`.
3. **Template + parameter hash** — structural `template_hash` *and* the exact
   `param_hash`; an L1 hit needs both.
4. **Temporal clock tick** *(temporal)* — logical time advances; λ update events
   may invalidate entries; τ-stale entries are evicted.
5. **Cache lookup** — **L1 exact** (template+param) hit returns the cached noisy
   answer for **free** (Δε=0); in semantic mode an **L2 similarity** hit can fire
   (⚠ similarity ≠ equivalence — may be a *wrong* answer, the honest negative).
6. **Sensitivity** — Δf (COUNT=1, SUM/AVG = column bound `B_c`).
7. **Allocate budget** — sequential composition `ε_total += ε`; `BudgetExhausted`
   → the query is **rejected**.
8. **Execute true query** — on the real Adult table (kept secret).
9. **Laplace mechanism** — `ε/n_aggs` per aggregate, noised at the recorded
   result position (GROUP BY = parallel composition).
10. **Cache write** + **11. Predictive bookkeeping** (re-estimate `Û`).

The right panel tracks the **budget ledger** (spent / total), **true vs.
released** values, and the **workload state**: distinct templates `u_k`, cache
size, savings `S(k)=1−u_k/k`, predicted `Û`, L2 semantic hits, and the temporal
clock / stale evictions. The bottom chart shows ε per query and the hit/miss
timeline (● miss · ○ L1 hit · ◑ L2 hit · ✕ reject).

## Modes (pick one — switching resets the session)

| Mode | ε per query | Cache | Extra steps shown |
|------|-------------|-------|-------------------|
| **exact** | — (no noise) | — | true answer, ε=0 (reference) |
| **naive** | fixed `ε_fixed` every query | **none** | budget burns down fast |
| **workload-aware** | fixed `ε_fixed`, on miss | L1 exact | savings from repeats |
| **semantic** | fixed `ε_fixed`, on miss | L1 + **L2 similarity** | the tree-kernel/embedding L2 (honest negative) |
| **temporal** | fixed `ε_fixed`, on miss | L1 + **staleness** | clock tick, τ eviction, λ updates, re-release cost |
| **predictive** | **`ε_q = B/Û`** (live) | L1 exact | the model adapting ε on the fly |

For **temporal** mode the `τ` (staleness tolerance) and `λ` (update rate) inputs
appear; small `τ` / large `λ` make cached answers expire and be re-released,
which you watch deplete the budget.

## Workload presets

The **Workloads** buttons run a sequence of queries so you watch the budget
deplete and cache hits accumulate:

- **W1 Repetitive** — the same dashboard tile 8× → after the first, all free.
- **W2 Parametric sweep** — age-band filters; repeats are free.
- **W4 Drill-down** — every query distinct → no savings (the model predicts this).

## Notes

- COUNT queries are the clean showcase (Δf=1, low noise). SUM/AVG use the
  conservative column-bound sensitivity `B_c`, so their noise is large by design
  — exactly the documented AVG/SUM limitation, not a bug.
- Adult categoricals carry a leading space (e.g. `sex = ' Male'`,
  `education = ' Doctorate'`).
- Single in-memory session (local single-user demo); **Reset** starts a fresh
  ledger/cache/allocator. No production code is modified — `demo/pipeline.py` is a
  thin instrumented orchestrator that calls the real components.
