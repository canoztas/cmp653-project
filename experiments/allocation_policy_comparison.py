"""Allocation-policy comparison: closed-form (ours) vs a bandit explorer vs naive.

Experimental counterpart to the BGTplanner positioning in the paper. BGTplanner
allocates a privacy budget online with a contextual bandit driven by *observed
utility* (validation accuracy in federated learning). In DP-SQL the per-query
error is NOT observable --- the true answer is exactly what privacy hides --- so
a utility-driven bandit cannot see its reward; the bandit here learns the
denominator from the only observable signals (granted eps, rejections). Policies
(end to end on Zipf workloads, cache misses consume budget):

  naive                 : eps_q = B/k  (fixed; prior fixed-eps systems).
  closed_form           : eps_q = B/U_hat, plug-in occupancy U_hat online.
  closed_form_safe      : eps_q = B/m  (OURS, the safe floor). Since u_k <= m,
                          total spend = (B/m)*u_k <= B -> never rejects, no
                          warmup. This IS the bandit's best-arm operating point
                          (d=m); the closed form jumps to it with no exploration.
  closed_form_rerelease : reserve + refresh worst cache entries (future-work fix).
  bandit                : epsilon-greedy over d in {k, k/2, k/4, m}, eps_q=B/d.
  modeldriven_bandit    : bandit over multipliers on U_hat (ours+theirs synthesis).
  oracle                : eps_q = B/u_k with the realised distinct count known
                          (ceiling; u_k is a public query-stream property).

Verified finding (adversarial multi-agent check, high confidence): at m=20,
k=100, B=10 the safe closed form answers 100% at fresh MAE ~2.0 vs the bandit's
~3.8 at the same 100% rate (~1.9x, ~18x SEM). This is an EXPLORATION-tax win, not
a noise-scale one: a bandit constrained to 100% answered only ties B/m. Crossing
BELOW B/m safely requires forecasting u_k (oracle ~1.6) -- which a model-free
bandit cannot do. Holds for m <= k/4; reverses for m > k/4 (bandit grid then has
a d<m arm). Report fresh MAE JOINTLY with %answered: sub-m bandit MAE is
rejection-bought. m and u_k are public (query stream), so all policies are
data-independent / DP-valid.

Run: python experiments/allocation_policy_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import expected_unique_queries, zipf_distribution
from dpdb.predictors import predict_smoothed_gt

M = 20
K = 100
B = 10.0          # total privacy budget
DELTA_F = 1.0     # COUNT sensitivity
FLOOR = 0.02
WARMUP = 5
ALPHA_GRID = [0.5, 1.0, 1.5, 2.0]
TRIALS = 60
SEED = 11


def _laplace_err(rng, eps):
    return abs(rng.laplace(0.0, DELTA_F / eps))


# Every policy returns (spent, answered, errs_all, errs_miss):
#   errs_miss = error on each FRESH release (isolates allocation quality)
#   errs_all  = errs_miss + errors inherited by cache hits (end-to-end utility)


def _u_hat_plugin(seen):
    counts = np.bincount(seen)
    counts = counts[counts > 0].astype(float) + 0.5
    p = counts / counts.sum()
    return expected_unique_queries(p, K)


def _u_hat_sgt(seen):
    counts = np.bincount(seen)
    counts = counts[counts > 0].astype(int).tolist()
    u = predict_smoothed_gt(counts, n=len(seen), k=K)
    return u if (u == u and u > 0) else float(len(counts))


def run_naive(rng, draws):
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache = {}
    eps = B / K
    for t in draws:
        if t in cache:
            all_e.append(cache[t]); answered += 1; continue
        if B - spent < max(eps, FLOOR):
            continue  # rejected
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1
    return spent, answered, all_e, miss_e


def _run_closed_form(rng, draws, u_hat_fn):
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache, seen = {}, []
    for i, t in enumerate(draws):
        if t in cache:
            all_e.append(cache[t]); answered += 1; seen.append(t); continue
        u_hat = K if len(seen) < 2 else u_hat_fn(seen)
        eps = (B / K) if i < WARMUP else (B / max(u_hat, 1.0))
        eps = max(FLOOR, min(eps, B - spent))
        if B - spent < FLOOR:
            seen.append(t); continue
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1; seen.append(t)
    return spent, answered, all_e, miss_e


def run_closed_form(rng, draws):
    return _run_closed_form(rng, draws, _u_hat_plugin)


def run_closed_form_sgt(rng, draws):
    """Closed form, but U_hat from Smoothed Good-Toulmin (Orlitsky-Suresh-Wu
    2016) instead of the under-predicting plug-in."""
    return _run_closed_form(rng, draws, _u_hat_sgt)


def run_oracle(rng, draws):
    """Ceiling of the closed-form *principle*: eps_q = B/u_k with the realized
    distinct count known exactly and no warmup. u_k (distinct query templates)
    is a property of the query stream, not of the protected database, so using
    it is DP-valid -- this is 'what if we predicted u_k perfectly'."""
    u_k = len(set(int(t) for t in draws))
    eps_star = B / max(u_k, 1)
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache = {}
    for t in draws:
        if t in cache:
            all_e.append(cache[t]); answered += 1; continue
        eps = min(eps_star, B - spent)
        if B - spent < FLOOR:
            continue
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1
    return spent, answered, all_e, miss_e


def run_closed_form_safe(rng, draws):
    """Closed-form allocation at the *safe* operating point eps_q = B/m, where
    m is the template-pool size (= max possible u_k). Because u_k <= m, total
    spend = (B/m)*u_k <= B, so this NEVER rejects and needs no warmup; the noise
    is uniform at scale Delta_f*m/B. This is exactly the operating point a grid
    bandit must *discover* by sampling (its d=m arm) -- the closed form jumps to
    it directly and skips the exploration tax. m is a property of the public
    query workload, not of the protected database."""
    eps = B / M
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache = {}
    for t in draws:
        if t in cache:
            all_e.append(cache[t]); answered += 1; continue
        if B - spent < eps - 1e-12:            # cannot happen while u_k <= m
            continue
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1
    return spent, answered, all_e, miss_e


def run_closed_form_rerelease(rng, draws, reserve=0.3):
    """Our documented 'cache re-release on budget surplus' fix, made online.

    Reserve a fraction of B; spend the rest on conservative first releases
    (eps_q=(1-reserve)*B/U_hat). After each release, spend from the reserve to
    *re-noise* the worst (lowest-eps) cache entry at the current quality, so the
    frequent warmup templates stop carrying their high warmup noise on every
    recurrence. Each re-release is a fresh independent eps-DP release; the
    ledger charges first + re-release eps (sequential composition), total <= B.
    A cache hit reads the *current* (possibly refreshed) noise."""
    first_budget = (1.0 - reserve) * B
    pool = reserve * B
    spent_first = 0.0
    answered, all_e, miss_e = 0, [], []
    cache = {}            # t -> current error
    eps_of = {}           # t -> eps currently backing the cache entry
    seen = []
    for i, t in enumerate(draws):
        if t in cache:
            all_e.append(cache[t]); answered += 1; seen.append(t); continue
        u_hat = K if len(seen) < 2 else _u_hat_plugin(seen)
        eps = (first_budget / K) if i < WARMUP else (first_budget / max(u_hat, 1.0))
        eps = max(FLOOR, min(eps, first_budget - spent_first))
        if first_budget - spent_first < FLOOR:
            seen.append(t); continue
        e = _laplace_err(rng, eps); spent_first += eps
        cache[t] = e; eps_of[t] = eps
        all_e.append(e); miss_e.append(e); answered += 1; seen.append(t)
        # re-release the worst cache entry from the reserve, at current quality
        if i >= WARMUP and pool >= eps and len(cache) > 1:
            worst = min(eps_of, key=lambda x: eps_of[x])
            if eps_of[worst] < eps:                  # only if it would improve
                cache[worst] = _laplace_err(rng, eps)
                eps_of[worst] = eps                  # latest release's quality
                pool -= eps
    return spent_first + (reserve * B - pool), answered, all_e, miss_e


def run_bandit(rng, draws, explore=0.2):
    """epsilon-greedy over denominators; observable reward only."""
    denoms = [K, K // 2, K // 4, M]
    q = [0.0] * len(denoms)   # mean reward estimate
    nsel = [0] * len(denoms)
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache = {}
    for t in draws:
        if t in cache:
            all_e.append(cache[t]); answered += 1; continue
        if rng.random() < explore or min(nsel) == 0:
            a = int(rng.integers(len(denoms)))
        else:
            a = int(np.argmax(q))
        eps = max(FLOOR, B / denoms[a])
        if B - spent < eps:                    # would exhaust -> rejection
            reward = -5.0                        # observable: a rejection happened
            nsel[a] += 1; q[a] += (reward - q[a]) / nsel[a]
            continue
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1
        reward = eps                             # observable: budget granted this query
        nsel[a] += 1; q[a] += (reward - q[a]) / nsel[a]
    return spent, answered, all_e, miss_e


def run_modeldriven_bandit(rng, draws, explore=0.2):
    """Synthesis (ours + theirs): an eps-greedy bandit whose arms are
    *multipliers on the closed-form U_hat*, eps_q = B/(U_hat*mult). The model
    supplies the operating point; the bandit only fine-tunes how tightly to
    spend around it. This is BGTplanner's adaptivity seeded with our forecast,
    and it needs no answer-key denominator grid."""
    mults = [0.7, 1.0, 1.4, 2.0]
    q = [0.0] * len(mults)
    nsel = [0] * len(mults)
    spent, answered, all_e, miss_e = 0.0, 0, [], []
    cache, seen = {}, []
    for i, t in enumerate(draws):
        if t in cache:
            all_e.append(cache[t]); answered += 1; seen.append(t); continue
        u_hat = K if len(seen) < 2 else _u_hat_plugin(seen)
        if rng.random() < explore or min(nsel) == 0:
            a = int(rng.integers(len(mults)))
        else:
            a = int(np.argmax(q))
        eps = max(FLOOR, B / (max(u_hat, 1.0) * mults[a]))
        if i < WARMUP:
            eps = max(FLOOR, B / M)        # safe warmup: u_k <= m
        if B - spent < eps:
            nsel[a] += 1; q[a] += (-5.0 - q[a]) / nsel[a]
            seen.append(t); continue
        e = _laplace_err(rng, eps); spent += eps
        cache[t] = e; all_e.append(e); miss_e.append(e); answered += 1; seen.append(t)
        nsel[a] += 1; q[a] += (eps - q[a]) / nsel[a]
    return spent, answered, all_e, miss_e


POLICIES = {
    "naive": run_naive,
    "closed_form": run_closed_form,
    "closed_form_sgt": run_closed_form_sgt,
    "closed_form_safe": run_closed_form_safe,
    "closed_form_rerelease": run_closed_form_rerelease,
    "bandit": run_bandit,
    "modeldriven_bandit": run_modeldriven_bandit,
    "oracle": run_oracle,
}
ORDER = ["naive", "closed_form", "closed_form_sgt", "closed_form_safe",
         "closed_form_rerelease", "bandit", "modeldriven_bandit", "oracle"]


def run():
    rng = np.random.default_rng(SEED)
    rows = []
    for alpha in ALPHA_GRID:
        p = zipf_distribution(M, alpha)
        agg = {name: {"mae": [], "mae_miss": [], "ans": [], "eps": []} for name in POLICIES}
        for _ in range(TRIALS):
            draws = rng.choice(M, size=K, p=p)
            for name, fn in POLICIES.items():
                spent, answered, all_e, miss_e = fn(rng, draws)
                agg[name]["mae"].append(np.mean(all_e) if all_e else np.nan)
                agg[name]["mae_miss"].append(np.mean(miss_e) if miss_e else np.nan)
                agg[name]["ans"].append(answered)
                agg[name]["eps"].append(spent)
        for name in POLICIES:
            mm = np.array(agg[name]["mae_miss"], dtype=float)
            sem = float(np.nanstd(mm) / np.sqrt(np.sum(~np.isnan(mm))))
            rows.append({
                "alpha": alpha, "policy": name,
                "mae": round(float(np.nanmean(agg[name]["mae"])), 2),
                "mae_miss": round(float(np.nanmean(mm)), 2),
                "mae_miss_sem": round(sem, 2),
                "answered": round(float(np.mean(agg[name]["ans"])), 1),
                "eps_used": round(float(np.mean(agg[name]["eps"])), 2),
            })
    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "predictors"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "allocation_policy_comparison.csv", index=False)

    print("\n=== Allocation policy comparison (M=20, K=100, B=10, avg over alpha) ===")
    print("  mae      = end-to-end (incl. cache hits inheriting early answers)")
    print("  mae_miss = fresh releases only (isolates allocation quality)\n")
    summ = df.groupby("policy").agg(mae=("mae", "mean"),
                                    mae_miss=("mae_miss", "mean"),
                                    mae_miss_sem=("mae_miss_sem", "mean"),
                                    answered=("answered", "mean"),
                                    eps_used=("eps_used", "mean")).round(2)
    summ = summ.reindex(ORDER)
    print(summ.to_string())
    print("\n(mae_miss_sem = mean per-alpha standard error; differences larger "
          "than ~2x the SEM are robust.)")
    print(f"\nSaved {out / 'allocation_policy_comparison.csv'}")
    return df


if __name__ == "__main__":
    run()
