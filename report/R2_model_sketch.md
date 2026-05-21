# R2 Analytical Model — Scratch Derivation

**Purpose.** Sketch the statistical model linking workload structure (template repetition skew), privacy budget, and utility — before touching the manuscript. Per the revision brief: confirm that the model recovers the toy `1 − 1/k` case at one limit and the naive `k` case at the other.

---

## Setup

Let:
- `m` = number of distinct query templates accessible to the analyst
- `{p_i}_{i=1..m}` = workload distribution over templates, Σ p_i = 1
- `k` = number of queries issued in a workload window
- `ε_q` = per-cache-miss privacy budget per release
- `Δf` = global sensitivity of a single template (e.g., 1 for COUNT)
- Each query `Q_t` (t = 1..k) is drawn i.i.d. from `{p_i}`
- Cache key = exact (template, parameter) identity → here, the template itself

The unique-query count after `k` requests:
```
u_k = | { i : at least one of Q_1, ..., Q_k equals template i } |
```

This is the **occupancy / coupon-collector** quantity for a non-uniform distribution.

---

## Proposition 1 — Expected unique queries

For each template i:
```
P(template i never appears in k draws) = (1 - p_i)^k
P(template i appears at least once)     = 1 - (1 - p_i)^k
```

By linearity of expectation:
```
E[u_k] = Σ_{i=1}^m [1 - (1 - p_i)^k]               (1)
```

This is exact under i.i.d. sampling. No closed form for arbitrary distribution, but always computable in O(m).

---

## Proposition 2 — Expected privacy budget consumption

Naive sequential composition (no caching, no parallel composition):
```
ε_naive(k) = k · ε_q                                (2)
```

Workload-aware exact-repeat caching (only first occurrence of each template costs ε_q;
all subsequent identical queries are post-processing of the cached output):
```
ε_wa(k) = u_k · ε_q                                 (3)

E[ε_wa(k)] = ε_q · E[u_k] = ε_q · Σ_i [1 - (1 - p_i)^k]   (4)
```

Budget savings ratio (in expectation):
```
S(k) = 1 - E[ε_wa(k)] / ε_naive(k)
     = 1 - (1/k) · Σ_i [1 - (1 - p_i)^k]            (5)
```

---

## Limit checks (the key test from the brief)

### Limit A — Perfect repetition (p_1 = 1, p_{i>1} = 0)

`p_1 = 1`, so `(1 - p_1)^k = 0` for all k ≥ 1, contributing `1` to (1).
`p_i = 0` for i > 1 contribute `0`.

```
E[u_k] = 1                                          ✓
E[ε_wa(k)] = ε_q
S(k) = 1 - 1/k                                      ✓ Recovers the toy formula
```

The professor's `1 − 1/k` (question-marked in the milestone) is the **degenerate corner case** of this model when the workload distribution collapses to a single template.

### Limit B — Uniform over m templates, m → ∞ with k fixed

`p_i = 1/m` for all i. Then:
```
(1 - 1/m)^k → 1 - k/m + O(1/m²)
E[u_k] = m · [1 - (1 - 1/m)^k] = m · [k/m - O(1/m)] → k    ✓
```

So every query is unique in expectation — recovers naive composition:
```
E[ε_wa(k)] → k · ε_q = ε_naive(k)
S(k) → 0                                             ✓
```

### Limit C — Truncated case: uniform over m, k → ∞ with m fixed

`(1 - 1/m)^k → 0` (k → ∞).
`E[u_k] = m · (1 - 0) = m`.
`E[ε_wa(k)] → m · ε_q` — *bounded* regardless of k.

This is the practically important case: a finite-template workload always has bounded budget under workload-aware accounting, but unbounded budget under naive composition.

**Budget exhaustion point under workload-aware:** k* is the first time u_k exceeds c = ⌊ε_total / ε_q⌋. If m ≤ c, k* = ∞ (workload never exhausts the budget). If m > c, k* is finite — a coupon-collector stopping time.

---

## Proposition 3 — Zipf(α) workload

Let p_i ∝ i^{-α} on {1, ..., m}, normalized by the generalized harmonic number:
```
H_{m,α} = Σ_{i=1}^m i^{-α}
p_i = i^{-α} / H_{m,α}
```

Asymptotics of H_{m,α}:
- α = 0:           H_{m,0} = m            (uniform, Limit B)
- 0 < α < 1:       H_{m,α} ≈ m^{1-α} / (1-α)
- α = 1:           H_{m,1} ≈ ln m + γ
- α > 1:           H_{m,α} → ζ(α) as m → ∞
- α → ∞:           p_1 → 1, all others → 0 (Limit A)

So Zipf(α) interpolates smoothly between Limit A (α → ∞) and Limit B (α → 0).

**Closed-form approximation for α > 1, large m:**
The top template has p_1 ≈ 1/ζ(α). Coupons with very small p_i are essentially never sampled, so the effective number of templates seen is bounded.

For α = 1.0 (the classic Zipf), p_1 ≈ 1/ln(m). For m = 10 templates:
```
H_{10,1} = 1 + 1/2 + ... + 1/10 ≈ 2.929
p_1 ≈ 0.341, p_2 ≈ 0.171, p_3 ≈ 0.114, ...
```
With k = 100 draws: E[u_k] = Σ_i [1 - (1 - p_i)^100]. Numerically ≈ 9.6 (almost all templates seen).
Without caching, ε_naive = 100·ε_q; with caching, ε_wa ≈ 9.6·ε_q. Savings ≈ 90%.

This will be computed exactly in the model code.

---

## Proposition 4 — Concentration of u_k

`u_k = f(Q_1, ..., Q_k)` is a function of k independent inputs, each affecting at most 1 unit of the output (changing one Q_t changes u_k by at most 1). By **McDiarmid's bounded-differences inequality**:
```
P(|u_k - E[u_k]| ≥ t) ≤ 2 exp(-2t² / k)              (6)
```

This gives an explicit tail bound on the budget exhaustion point:
```
P(ε_wa(k) ≥ c · ε_q) = P(u_k ≥ c)
                    ≤ exp(-2(c - E[u_k])² / k)   when c > E[u_k]
```

So the workload-aware budget consumption concentrates within `±O(√k)` around `E[u_k]`, regardless of the workload distribution.

---

## Proposition 5 — Utility model

For a unique cache-miss query, expected absolute error is the Laplace mean:
```
E[|η|] = b = Δf / ε_q                                (7)
Var(η) = 2 b²                                       
```

For a repeated cache-hit query, the error is **identical to the first observation** (deterministic replay). The marginal error distribution per query is the same `Lap(b)`; the difference is the **joint** distribution:
- Naive: k independent draws, so averaging gives Var = 2b²/k.
- Workload-aware: u_k independent draws, replayed; averaging the same draws does not reduce variance.

This is the key **caveat** the milestone glossed over: workload-aware saves budget but does not let the analyst denoise via averaging. The trade-off is honest to surface.

Under a **fixed total budget** ε_total, the per-query budget the system can afford is:
- Naive: ε_q^naive = ε_total / k
- Workload-aware: ε_q^wa = ε_total / E[u_k]    (split across unique queries only)

Per-query expected error:
- Naive: Δf · k / ε_total
- Workload-aware: Δf · E[u_k] / ε_total = Δf / ε_q^wa

**Predicted utility gain** for the same total budget:
```
error_naive / error_wa = k / E[u_k]                  (8)
```

For Zipf(1.0), m=10, k=100: ratio ≈ 100/9.6 ≈ 10×. The middleware can answer the same workload with ~10× less per-query error at the same privacy cost.

---

## Proposition 6 — Temporal extension (R3 preview)

Let updates arrive as a Poisson process with rate λ. Each cached entry has a validity window of staleness tolerance τ. The expected number of re-noisings for template i over time horizon T is:
```
N_i(T) = ⌈T · λ / (templates affected per update)⌉ + ⌈T / τ⌉   (worst case)
```

For the simplest **bounded-staleness** model with no updates between refreshes:
```
N_i(T) = ⌈T / τ⌉                                     
```

Total expected budget over horizon T:
```
E[ε_wa(T)] = E[u_∞] · ⌈T / τ⌉ · ε_q                 (9)
```

Compare to naive:
```
E[ε_naive(T)] = (queries arrived in T) · ε_q
```

As τ → ∞ (static snapshot), formula (9) reduces to E[u_∞] · ε_q, recovering the original model. As τ → 0, every query is a fresh release — recovering naive composition. The temporal axis is a continuous slider between the two regimes.

---

## Summary of what the model gives us

| Knob | Effect on E[ε_wa] | Interpretation |
|------|-------------------|----------------|
| α ↑ (more skew) | ↓ | Heavier repetition → more caching wins |
| m ↑ (more templates) | ↑ | Diverse workload → less reuse |
| k ↑ (longer workload) | ↑ (sublinearly for non-trivial α) | More queries see more templates |
| τ ↓ (fresher data) | ↑ | Caches expire, must re-noise |
| λ ↑ (more updates) | ↑ | Invalidations trigger re-noising |

**Predictions for experimental validation (R6):**
- W1 (perfect repetition, α=∞): savings = 1 − 1/k. With k=100, savings = 99%.
- W2 (Zipf parametric, α≈1.0 over m=5): savings ≈ 90% (most queries are repeats).
- W3 (diverse): savings near 0 unless internal repetition exists.
- W4 (drill-down): savings = 0 (each query strictly narrower → no exact repeats).
- Adding temporal regime with τ=100 queries: savings degrade by factor ⌈k/τ⌉.

These will be the validation plots: model curve (analytical) vs empirical mean ± 95% CI.

---

## Open questions resolved

- **"What is the use of caching in DP?"** (Abstract comment): The answer is now the analytical model — caching is a mechanism instantiation of a deeper structural fact, namely that workloads have low intrinsic privacy dimension when E[u_k] ≪ k. The contribution becomes the model that predicts when and by how much this holds.
- **"1 − 1/k?"** (Section 3 question mark): Limit A of equation (5). It is the corner case, not the headline.
- **"Algorithm 1 trivial!"**: New Algorithm 2 will include cache validity check, temporal re-noising trigger, and ledger update consistent with the model.
- **"AVG — why?"**: Because the analytical model needs ε_q-per-component accounting to predict the 2g·ε_q cost of a GROUP BY AVG. The SUM/COUNT split makes this visible and predictable.
- **"Caching is already working like that"**: True — and the milestone's mistake was claiming caching as the contribution. The model is the contribution; caching is the mechanism it predicts.
