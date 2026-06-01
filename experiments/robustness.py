"""Robustness evidence for the occupancy forecast E[u_k]=sum(1-(1-p_i)^k).

Two questions:

(1) DISTRIBUTION-AGNOSTIC (sanity). The closed form is exact for ANY i.i.d.
    marginal, not just Zipf. We verify it on uniform, Zipf, and a lognormal-ish
    marginal: the 30-trial empirical mean must match E[u_k] (it does, by
    construction -- this is a Monte-Carlo self-consistency check, not external
    validity, and we label it as such).

(2) NON-I.I.D. STRESS (the meaningful test). Real workloads are bursty: a
    template, once issued, tends to repeat. We model this with a "sticky" Markov
    process: stay on the previous template with probability s, else draw fresh
    from the same Zipf marginal (so the stationary marginal is unchanged). As s
    grows, distinct templates u_k DROP below the i.i.d. forecast -- i.e. the
    forecast OVER-predicts the budget the workload spends. That is exactly the
    safe direction: the i.i.d. forecast is a conservative upper bound under
    positive correlation, so the privacy-budget guarantee is not violated by
    burstiness; only the savings are under-counted.

Deterministic seeds. Run: python experiments/robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dpdb.model import expected_unique_queries, occupancy_variance, zipf_distribution

M, K, TRIALS = 20, 100, 200


def _lognormalish(m, sigma=1.0):
    x = np.exp(sigma * np.arange(m) / m)[::-1]
    return x / x.sum()


def _draw_iid(rng, p, k):
    return rng.choice(len(p), size=k, p=p)


def _draw_sticky(rng, p, k, s):
    out = np.empty(k, dtype=int)
    out[0] = rng.choice(len(p), p=p)
    for t in range(1, k):
        out[t] = out[t - 1] if rng.random() < s else rng.choice(len(p), p=p)
    return out


def run():
    rng = np.random.default_rng(7)
    print("=== (1) distribution-agnostic: E[u_k] is exact for any i.i.d. marginal ===")
    dists = {"uniform": np.full(M, 1 / M),
             "zipf(1)": zipf_distribution(M, 1.0),
             "lognormal-ish": _lognormalish(M)}
    for name, p in dists.items():
        pred = expected_unique_queries(p, K)
        emp = np.mean([len(set(_draw_iid(rng, p, K).tolist())) for _ in range(TRIALS)])
        sd = occupancy_variance(p, K) ** 0.5
        print(f"  {name:14s} E[u_k]={pred:5.2f}  empirical={emp:5.2f}  "
              f"|err|={abs(pred-emp):.2f}  (forecast std sqrt(V)={sd:.2f})")

    print("\n=== (2) non-i.i.d. burstiness: i.i.d. forecast OVER-predicts (conservative) ===")
    p = zipf_distribution(M, 1.0)
    pred = expected_unique_queries(p, K)
    print(f"  Zipf(1), m={M}, k={K}: i.i.d. forecast E[u_k] = {pred:.2f}")
    print(f"  {'stickiness s':>12} | {'empirical u_k':>13} | {'forecast/empirical':>18} | direction")
    for s in (0.0, 0.3, 0.6, 0.9):
        emp = np.mean([len(set(_draw_sticky(rng, p, K, s).tolist())) for _ in range(TRIALS)])
        ratio = pred / emp
        d = "exact (i.i.d.)" if s == 0 else ("OVER-predicts -> SAFE" if pred >= emp else "UNDER -> unsafe")
        print(f"  {s:12.1f} | {emp:13.2f} | {ratio:17.2f}x | {d}")
    print("\n  Burstiness reduces distinct templates, so the i.i.d. forecast is an "
          "upper bound on\n  realized u_k -> the eps_q*E[u_k] budget estimate stays "
          "conservative (never under-spends).")


if __name__ == "__main__":
    run()
