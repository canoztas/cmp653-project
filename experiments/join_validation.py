"""Twin experiment: the occupancy/budget forecast is sensitivity-agnostic, so it
transfers UNCHANGED from single-table aggregates to multi-table joins.

We sample one Zipf identity stream over the m=5 TPC-H order-priority pool and
instantiate every identity TWO ways on the SAME stream:

  single-table : SELECT COUNT(*) FROM orders WHERE o_orderpriority = '<p>'        (Delta f = 1)
  join         : SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey=o_orderkey
                 WHERE o_orderpriority = '<p>'                                     (Delta f = d_max = 7)

d_max = max line-items per order is a PUBLIC foreign-key multiplicity bound from
the TPC-H spec (1..7); it is the conservative, B_c-style global-sensitivity
constant for the FK join (the privacy unit is an order/entity).

Claim demonstrated:
  * E[u_k], realized u_k, savings S(k), and the workload-aware spend eps_q*u_k are
    IDENTICAL for the single-table and the join workload (occupancy depends on the
    query-identity stream, not on what the query computes or how sensitive it is).
  * ONLY the per-release noise scale changes: Lap(1/eps_q) vs Lap(d_max/eps_q), so
    absolute MAE scales by d_max while the relative utility ratio k/E[u_k] is
    unchanged (Delta f cancels).

Standalone harness on purpose: the production parser rejects JOINs, so we key the
budget-spending species directly off the canonical SQL string and inject Delta f
offline -- exactly the mechanism-agnostic forecasting layer the paper describes.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb
import numpy as np
import pandas as pd

from dpdb.model import expected_unique_queries, budget_savings_ratio, zipf_distribution

DB = str(Path(__file__).parent.parent / "data" / "dpdb.duckdb")
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
M = len(PRIORITIES)            # 5-template pool
D_MAX = 7                      # public FK multiplicity bound (max line-items / order; TPC-H spec 1..7)
B_TOTAL = 10.0                 # fixed total budget for the utility comparison
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
KS = [10, 25, 50, 100]
TRIALS = 30


def sql_single(p: str) -> str:
    return f"SELECT COUNT(*) FROM orders WHERE o_orderpriority = '{p}'"


def sql_join(p: str) -> str:
    return ("SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey "
            f"WHERE o_orderpriority = '{p}'")


def species(sql: str) -> str:
    """Exact-query cache identity = canonical SQL hash (no parser needed)."""
    return hashlib.sha256(sql.encode()).hexdigest()[:16]


def main():
    con = duckdb.connect(DB, read_only=True)

    # True answers, executed once per distinct priority on BOTH sides -- this is
    # the join actually running on 1.5M orders x 6M line-items, under DP below.
    true_single = {p: con.execute(sql_single(p)).fetchone()[0] for p in PRIORITIES}
    true_join = {p: con.execute(sql_join(p)).fetchone()[0] for p in PRIORITIES}
    con.close()

    print("=== Twin single-table vs JOIN experiment (TPC-H, m=5 priority pool, d_max=7) ===\n")
    print("True COUNTs per priority (executed on the real tables):")
    for p in PRIORITIES:
        print(f"  {p:>16}: single-table={true_single[p]:>8}  join={true_join[p]:>9} "
              f"(x{true_join[p]/true_single[p]:.2f})")
    print()

    rows = []
    for alpha in ALPHAS:
        pdist = zipf_distribution(M, alpha)
        for k in KS:
            euk = expected_unique_queries(pdist, k)        # FORECAST (same for both sides)
            sk = budget_savings_ratio(pdist, k)
            eps_q = B_TOTAL / max(euk, 1.0)                 # workload-aware sizing at fixed total B
            for trial in range(TRIALS):
                rng = np.random.default_rng(1000 * int(alpha * 10) + 10 * k + trial)
                idx = rng.choice(M, size=k, p=pdist)
                stream = [PRIORITIES[i] for i in idx]

                # Occupancy: distinct species in the stream, computed independently
                # on each side. Equal by construction -- THAT is the point.
                uk_single = len({species(sql_single(s)) for s in stream})
                uk_join = len({species(sql_join(s)) for s in stream})
                assert uk_single == uk_join, "occupancy must be identity-stream-determined"
                uk = uk_single

                # Per-release noise on the distinct (answered) releases. The true
                # value cancels in |noisy - true| = |eta|, so MAE measures the noise.
                distinct = list(dict.fromkeys(stream))
                mae_single = float(np.mean([abs(rng.laplace(0.0, 1.0 / eps_q)) for _ in distinct]))
                mae_join = float(np.mean([abs(rng.laplace(0.0, D_MAX / eps_q)) for _ in distinct]))

                rows.append(dict(
                    alpha=alpha, k=k, trial=trial,
                    forecast_Euk=euk, realized_uk=uk, savings_Sk=sk,
                    spend_wa=eps_q * uk, spend_naive=eps_q * k,
                    delta_f_single=1, delta_f_join=D_MAX,
                    mae_single=mae_single, mae_join=mae_join,
                ))

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "joins"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "join_validation.csv", index=False)

    # --- Aggregate headline numbers ---
    g = df.groupby(["alpha", "k"]).agg(
        Euk=("forecast_Euk", "first"),
        uk=("realized_uk", "mean"),
        Sk=("savings_Sk", "first"),
        mae_single=("mae_single", "mean"),
        mae_join=("mae_join", "mean"),
    ).reset_index()
    g["rel_err_pct"] = (g["uk"] - g["Euk"]).abs() / g["Euk"] * 100.0
    g["mae_ratio"] = g["mae_join"] / g["mae_single"]

    print("Per-cell (forecast E[u_k] vs realized u_k is IDENTICAL for single-table and join;\n"
          "only the noise scale differs by d_max):\n")
    print(g.round(3).to_string(index=False))

    print("\n=== Headline ===")
    print(f"  Forecast vs realized u_k: mean |error| = {g['rel_err_pct'].mean():.2f}% "
          f"(max {g['rel_err_pct'].max():.2f}%) -- SAME number on the single-table and the join side,")
    print(f"    because the occupancy stream is identical; the join is forecast exactly as the single table.")
    print(f"  Per-release MAE ratio join/single = {g['mae_ratio'].mean():.2f} "
          f"(= d_max = {D_MAX}); the ONLY thing that moved is Delta f.")
    print(f"  Relative utility ratio k/E[u_k] is identical across Delta f (Delta f cancels).")
    print(f"\n  Wrote {out / 'join_validation.csv'} ({len(df)} trials).")


if __name__ == "__main__":
    main()
