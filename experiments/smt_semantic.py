"""End-to-end demo: an SMT-backed equivalence cache reuses noisy answers across
queries the *syntactic* normaliser cannot prove equal, soundly and free.

The professor's S5 suggestion was to use "symbolic solvers" (e.g. to see that
age>=30 == age>29). The decidable normaliser already gets that case; here we go
further, with Z3 deciding equivalences it cannot: integer arithmetic
(age*2>=60 == age>=30), disjunctions of ranges, and De Morgan. The matcher
admits a hit ONLY when Z3 *proves* equivalence, so there are no wrong reuses;
non-equivalent queries (age>=31) are correctly refused.

Z3 is an optional dependency. Run: python experiments/smt_semantic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb

from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.smt_equiv import SMTEquivalenceMatcher, z3_available

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
    if not z3_available():
        print("z3 is not installed; SMT semantic cache is disabled (pip install z3-solver).")
        return
    cfg = Config.from_yaml(str(Path(__file__).parent.parent / "config.yaml"))
    int_cols = integer_columns("adult")
    mw = DPMiddleware(cfg, mode=ExecutionMode.SEMANTIC_DP)
    mw.budget.semantic_matcher = SMTEquivalenceMatcher(int_cols)

    print("=== SMT-backed equivalence cache (LIVE middleware) ===\n")
    scenario = [
        ("age >= 30",                          "fresh: first release"),
        ("age*2 >= 60",                        "EQUIVALENT (arithmetic) -> free hit"),
        ("age >= 30 OR age >= 40",             "EQUIVALENT (OR of ranges) -> free hit"),
        ("NOT (age < 30)",                     "EQUIVALENT (negation) -> free hit"),
        ("age >= 31",                          "NOT equivalent -> spends budget"),
    ]
    print(f"  {'query':<26} | {'eps':>4} | {'hit?':>5} | note")
    spent = 0.0
    hits = 0
    for where, note in scenario:
        r = mw.execute(f"SELECT COUNT(*) FROM adult WHERE {where}", epsilon=1.0)
        spent += r.epsilon_used
        hits += r.cache_hit
        print(f"  {where:<26} | {r.epsilon_used:>4.1f} | {str(r.cache_hit):>5} | {note}")

    print(f"\n  total spent = {spent:.1f} over {len(scenario)} queries; {hits} free SMT hits.")
    ok = (spent == 2.0 and hits == 3)
    print(f"  expected 2 paid / 3 free (sound): {ok}")
    print("  The normaliser alone would charge all of these (different ASTs); Z3 proves\n"
          "  the three equivalences and refuses age>=31 -- no wrong answer is ever reused.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
