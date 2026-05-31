"""Allocation-policy comparison: closed-form (ours) vs a bandit explorer vs naive.

This is the experimental counterpart to the BGTplanner positioning in the paper.
BGTplanner allocates a privacy budget online with a contextual bandit driven by
*observed utility* (validation accuracy in federated learning). In a DP-SQL
setting the per-query error is NOT observable --- the true answer is exactly
what privacy hides --- so a utility-driven bandit cannot see its reward. We
therefore compare three online allocation policies that use only *observable*
signals, end to end on Zipf workloads:

  naive       : eps_q = B/k  (fixed, what prior fixed-eps systems do).
  closed_form : eps_q = B/U_hat  (ours; U_hat from the occupancy model online).
  bandit      : epsilon-greedy over denominators d in {k, k/2, k/4, m};
                eps_q = B/d; reward = eps_q on success, large penalty on a
                budget-exhausted rejection (both observable). A BGTplanner-style
                *approximation* --- it learns the denominator online instead of
                using the closed-form U_hat.

Metric of interest: mean absolute error over answered queries, the fraction of
queries answered, and budget used. Deterministic seeds.

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


POLICIES = {
    "naive": run_naive,
    "closed_form": run_closed_form,
    "closed_form_sgt": run_closed_form_sgt,
    "bandit": run_bandit,
}
ORDER = ["naive", "closed_form", "closed_form_sgt", "bandit"]


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
            rows.append({
                "alpha": alpha, "policy": name,
                "mae": round(float(np.nanmean(agg[name]["mae"])), 2),
                "mae_miss": round(float(np.nanmean(agg[name]["mae_miss"])), 2),
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
                                    answered=("answered", "mean"),
                                    eps_used=("eps_used", "mean")).round(2)
    summ = summ.reindex(ORDER)
    print(summ.to_string())
    print(f"\nSaved {out / 'allocation_policy_comparison.csv'}")
    return df


if __name__ == "__main__":
    run()
