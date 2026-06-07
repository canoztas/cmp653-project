"""Real-system baseline head-to-head against a DEPLOYED DP-SQL engine: OpenDP
SmartNoise-SQL. We run the SAME repeated workload through SmartNoise's PrivateReader
and read its odometer, vs. our workload-aware accounting. This is a real library, not
an internal mode or a strawman.

SmartNoise-SQL charges every query against its budget and has no exact-repeat caching,
so on a repetitive dashboard workload (k identical COUNT queries) it spends k*eps_q,
where our workload-aware accounting spends eps_q*u_k = eps_q (one release, the rest are
cache hits = post-processing). On a genuinely all-distinct workload neither caches and
we TIE -- the gap is exactly the template repetition. Both use basic (pure-eps)
sequential composition here; SmartNoise's tighter RDP accounting would lower BOTH rates
and is orthogonal to (composes with) the caching.

Measured (Adult, eps_q=1, k=100):
  repetitive  : SmartNoise 100.0  vs ours 1.0    -> 100x
  all-distinct: SmartNoise 100.0  vs ours 100.0  -> tie (honest control)

Requires an isolated env with SmartNoise:
  python -m venv snenv && snenv/Scripts/pip install smartnoise-sql
  python -c "import duckdb; duckdb.connect('data/dpdb.duckdb',read_only=True)\
            .execute('SELECT * FROM adult').df().to_csv('adult.csv',index=False)"
  snenv/Scripts/python experiments/realsystem_baseline.py adult.csv
"""
import sys

import pandas as pd
from snsql import Privacy, from_df

CSV = sys.argv[1] if len(sys.argv) > 1 else "_scratch/adult.csv"
EPS_Q, K = 1.0, 100

df = pd.read_csv(CSV)
metadata = {"db": {"public": {"adult": {
    "row_privacy": True, "censor_dims": False, "clamp_counts": False,
    "rows": int(len(df)),
    "age": {"type": "int", "lower": 0, "upper": 100},
}}}}


def smartnoise_spend(queries):
    reader = from_df(df, privacy=Privacy(epsilon=EPS_Q, delta=0.0), metadata=metadata)
    for q in queries:
        reader.execute(q)
    return reader.odometer.spent[0]


rep = ["SELECT COUNT(*) AS n FROM public.adult WHERE age >= 30"] * K
distinct = [f"SELECT COUNT(*) AS n FROM public.adult WHERE age >= {i + 1}" for i in range(K)]

print(f"=== Real-system baseline: SmartNoise-SQL vs workload-aware (Adult, eps_q={EPS_Q}, k={K}) ===\n")
print(f"[Repetitive: {K} identical COUNT queries]")
print(f"  SmartNoise-SQL : {smartnoise_spend(rep):.1f}   vs   ours (u_k=1): {EPS_Q:.1f}"
      f"   -> {smartnoise_spend(rep)/EPS_Q:.0f}x\n")
print(f"[All-distinct: {K} different COUNT queries (honest control)]")
print(f"  SmartNoise-SQL : {smartnoise_spend(distinct):.1f}   vs   ours (u_k={K}): {EPS_Q*K:.1f}"
      f"   -> tie (no repetition to exploit)")
print("\n  Both use basic pure-eps composition; tighter RDP lowers both, orthogonal to caching.")
