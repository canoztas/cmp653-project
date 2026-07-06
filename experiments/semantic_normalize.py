"""End-to-end demo: a SOUND equivalence-checked cache saves budget on queries the
exact cache misses, with zero risk of a wrong-answer reuse.

The paper reports a NEGATIVE result for similarity-based semantic caching (tree
kernel + embedding): it admits false positives and misses textbook equivalences.
This demo realises the safe alternative from the same section -- predicate
normalisation -- and runs it through the LIVE middleware. We swap the unsound
similarity matcher for the sound NormalizedEquivalenceMatcher (no change to the
budget code: it uses the same SEMANTIC_AWARE plug point) and show:

  * age > 29  and  age >= 30          -> the second is a FREE cache hit (eps=0),
  * sex='M' AND age>=30  reordered     -> also a free hit (AND commutativity),
  * age >= 31                          -> NOT matched, spends budget (soundness).

Integer-typed columns are read from the schema (a public property), so the
>/>= tightening is applied only where it is provably valid.

Run: python experiments/semantic_normalize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.normalize import NormalizedEquivalenceMatcher

DB = str(Path(__file__).parent.parent / "data" / "dpdb.duckdb")


def integer_columns(table: str) -> set:
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND data_type IN ('INTEGER','BIGINT','SMALLINT','HUGEINT')",
        [table]).fetchall()
    con.close()
    return {r[0] for r in rows}


def main():
    cfg = Config.from_yaml(str(Path(__file__).parent.parent / "config.yaml"))
    int_cols = integer_columns("adult")
    print(f"=== Sound equivalence-checked cache (LIVE middleware) ===")
    print(f"    integer columns from schema: {sorted(int_cols)}\n")

    mw = DPMiddleware(cfg, mode=ExecutionMode.SEMANTIC_DP)
    # Replace the unsound similarity matcher with the sound normaliser.
    mw.budget.semantic_matcher = NormalizedEquivalenceMatcher(int_cols)

    scenario = [
        ("age > 29",                 "fresh: first release, spends budget"),
        ("age >= 30",                "EQUIVALENT to age>29 -> free hit"),
        ("sex = 'M' AND age >= 30",  "fresh: adds a predicate"),
        ("age > 29 AND sex = 'M'",   "EQUIVALENT (tighten + AND-reorder) -> free hit"),
        ("age >= 31",                "NOT equivalent (age>=31) -> spends budget"),
    ]

    print(f"  {'query':<28} | {'eps':>4} | {'hit?':>5} | note")
    spent = 0.0
    hits = 0
    for where, note in scenario:
        sql = f"SELECT COUNT(*) FROM adult WHERE {where}"
        r = mw.execute(sql, epsilon=1.0)
        spent += r.epsilon_used
        hits += r.cache_hit
        print(f"  {where:<28} | {r.epsilon_used:>4.1f} | {str(r.cache_hit):>5} | {note}")

    print(f"\n  total spent = {spent:.1f} over {len(scenario)} queries; "
          f"{hits} free equivalence hits.")
    # 2 of the 5 are provable equivalents of an earlier query -> 3 paid, 2 free.
    ok = (spent == 3.0 and hits == 2)
    print(f"  expected 3 paid / 2 free (sound): {ok}")
    print(f"\n  The exact cache alone would have charged all 5 (different templates); "
          f"the\n  normalised cache saves the 2 provable equivalents and -- crucially -- "
          f"does NOT\n  match age>=31, so no wrong answer is ever reused.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
