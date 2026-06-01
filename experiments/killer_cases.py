"""Killer cases: workloads where the naive-composition baseline fails badly and
the workload-aware / predictive model wins -- run on the REAL middleware + Adult
DB, and reported honestly (the drill-down case is included precisely because we do
NOT win there).

Two distinct naive failure modes:
  (A) fixed eps_q  -> naive EXHAUSTS the budget after B/eps_q releases and REJECTS
      the rest; the cache answers every exact repeat for free, so it answers all.
  (B) eps_q = B/k  -> naive spreads the budget thin (tiny eps per query, huge
      noise); the model concentrates it on the u_k distinct queries (eps_q=B/u_k),
      cutting per-query error by ~k/u_k.

Metric: fraction answered and mean |error| over answered queries. Deterministic.
Run: python experiments/killer_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode

TILE = "SELECT COUNT(*) FROM adult WHERE age >= 30"          # one dashboard KPI
TRUE = None  # filled from EXACT
K, B, TRIALS = 100, 10.0, 12


def _true(sql):
    cfg = Config.from_yaml()
    return float(DPMiddleware(cfg, mode=ExecutionMode.EXACT).execute(sql).rows[0][0])


def _run(mode, queries, eps_q, seed):
    cfg = Config.from_yaml()
    cfg.privacy.total_epsilon = B
    cfg.privacy.default_query_epsilon = eps_q
    np.random.seed(seed)
    mw = DPMiddleware(cfg, mode=mode, predictive_k_total=len(queries),
                      predictive_warmup_fraction=0.05)
    answered, errs = 0, []
    for q in queries:
        r = mw.execute(q, epsilon=eps_q)
        if r.error or not r.rows:
            continue
        answered += 1
        errs.append(abs(float(r.rows[0][0]) - TRUE[q]))
    return answered, (float(np.mean(errs)) if errs else float("nan"))


def _avg(mode, queries, eps_q):
    a, e = [], []
    for t in range(TRIALS):
        ans, err = _run(mode, queries, eps_q, seed=t * 17 + 1)
        a.append(ans); e.append(err)
    return np.mean(a), np.nanmean(e)


def run():
    global TRUE
    repetitive = [TILE] * K
    drill = [f"SELECT COUNT(*) FROM adult WHERE age >= {a}" for a in range(K)]  # all unique
    TRUE = {q: _true(q) for q in set(repetitive + drill)}

    print(f"=== Killer cases (real middleware + Adult, B={B:g}, k={K}, {TRIALS} trials) ===\n")

    print("CASE A -- repetitive dashboard, fixed eps_q=1.0 (naive runs out of budget):")
    na, ne = _avg(ExecutionMode.NAIVE_DP, repetitive, 1.0)
    wa, we = _avg(ExecutionMode.WORKLOAD_DP, repetitive, 1.0)
    print(f"  NAIVE      : answered {na:5.1f}/{K}   |err| {ne:6.2f}")
    print(f"  WORKLOAD   : answered {wa:5.1f}/{K}   |err| {we:6.2f}")
    print(f"  -> we answer {wa/na:.0f}x more queries (cache frees the repeats)\n")

    print("CASE B -- repetitive dashboard, fixed TOTAL budget B (accuracy):")
    na2, ne2 = _avg(ExecutionMode.NAIVE_DP, repetitive, B / K)        # eps_q=B/k, spread thin
    wa2, we2 = _avg(ExecutionMode.WORKLOAD_DP, repetitive, B)          # eps_q=B on the 1 distinct
    print(f"  NAIVE (eps_q=B/k={B/K:.2f}) : answered {na2:5.1f}/{K}   |err| {ne2:6.2f}")
    print(f"  WORKLOAD (eps_q=B/u_k)      : answered {wa2:5.1f}/{K}   |err| {we2:6.2f}")
    ratio = "essentially exact" if we2 < 0.5 else f"{ne2/we2:.0f}x less error"
    print(f"  -> naive's |err|~{ne2:.0f} vs {ratio} (whole budget B on the u_k=1 distinct query)\n")

    print("CASE C (honest counter) -- investigative drill-down, all queries unique:")
    nc, nce = _avg(ExecutionMode.NAIVE_DP, drill, B / K)
    wc, wce = _avg(ExecutionMode.WORKLOAD_DP, drill, B / K)
    print(f"  NAIVE      : answered {nc:5.1f}/{K}   |err| {nce:6.2f}")
    print(f"  WORKLOAD   : answered {wc:5.1f}/{K}   |err| {wce:6.2f}")
    print(f"  -> S(k)=0, no repeats: we match naive (the model predicts this, no overclaim).")


if __name__ == "__main__":
    run()
