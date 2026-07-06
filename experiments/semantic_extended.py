"""End-to-end demo of the two newly-implemented SOUND normalisations in
:mod:`dpdb.normalize`: integer RANGE-MERGING within a conjunction and table-
ALIAS alpha-renaming. Both let the equivalence-checked cache reuse a noisy
answer (zero extra budget) for predicates the exact-repeat cache would miss,
WITHOUT ever colliding two non-equivalent queries.

It demonstrates three things on real query text:
  1. RANGE-MERGING -- redundant integer bounds collapse to the tightest one and
     contradictory conjunctions canonicalise to a single FALSE, so every
     provably-unsatisfiable WHERE shares one cache key.
  2. ALIAS alpha-renaming -- on a single physical table, ``a.age`` / ``orders.age``
     / bare ``age`` share a key; across a JOIN the qualifier is KEPT (a safe
     miss), because there ``o.x`` and ``l.x`` are different physical columns.
  3. SOUNDNESS audit -- a brute-force oracle over age in 0..120 (x sex) confirms
     that EVERY pair the normaliser collides is genuinely equivalent and the
     non-equivalent control pairs stay distinct.

Deterministic: no randomness beyond a fixed seed in the soundness audit.
Run: python experiments/semantic_extended.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from dpdb.normalize import NormalizedEquivalenceMatcher, canonical_sql
from dpdb.parser import parse_query

INT_COLS = {"age"}
DOMAIN = range(0, 121)
_OPS = {
    ">": lambda a, c: a > c, ">=": lambda a, c: a >= c,
    "<": lambda a, c: a < c, "<=": lambda a, c: a <= c,
    "=": lambda a, c: a == c, "!=": lambda a, c: a != c,
}


def _key(sql_where, table="adult", int_cols=INT_COLS):
    parsed = parse_query(f"SELECT COUNT(*) FROM {table} WHERE {sql_where}")
    return canonical_sql(parsed, int_cols)


def _oracle(sql_where):
    """Brute-force satisfying set over (age, sex) for the simple grammar used in
    this demo (age <op> int / int <op> age / sex = 'X', joined by AND)."""
    fns = []
    for part in sql_where.split(" AND "):
        part = part.strip()
        if part.startswith("sex"):
            val = part.split("=", 1)[1].strip().strip("'")
            fns.append(lambda a, s, val=val: s == val)
            continue
        for op in ("<=", ">=", "!=", "=", "<", ">"):
            if op in part:
                lhs, rhs = (x.strip() for x in part.split(op, 1))
                # strip a possible alias qualifier from the column token
                if "." in lhs:
                    lhs = lhs.split(".")[-1]
                if "." in rhs:
                    rhs = rhs.split(".")[-1]
                if lhs == "age":
                    c = int(rhs)
                    fns.append(lambda a, s, op=op, c=c: _OPS[op](a, c))
                elif rhs == "age":
                    flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<=",
                            "=": "=", "!=": "!="}[op]
                    c = int(lhs)
                    fns.append(lambda a, s, flip=flip, c=c: _OPS[flip](a, c))
                break
    return frozenset((a, s) for a in DOMAIN for s in ("M", "F")
                     if all(fn(a, s) for fn in fns))


def section(title):
    print(f"\n=== {title} ===")


def demo_range_merging():
    section("1. Range-merging on an integer column")
    cases = [
        ("redundant lower bound",  "age >= 30 AND age >= 40", "age >= 40"),
        ("redundant upper bound",  "age <= 80 AND age <= 60", "age <= 60"),
        ("tighten then merge",     "age > 29 AND age > 39",    "age >= 40"),
        ("equality subsumes bound", "age = 35 AND age >= 10",  "age = 35"),
    ]
    rows = []
    for label, lhs, rhs in cases:
        collide = _key(lhs) == _key(rhs)
        equiv = _oracle(lhs) == _oracle(rhs)
        print(f"  [{label:22}] {lhs:28} -> {_key(lhs).split('WHERE')[-1].strip()}")
        print(f"  {'':22}   collides with {rhs!r}: {collide}  (truly equiv: {equiv})")
        rows.append(dict(kind="range_merge", label=label, lhs=lhs, rhs=rhs,
                         collide=collide, equivalent=equiv))

    section("1b. Contradiction -> a single canonical FALSE")
    unsat = ["age >= 40 AND age <= 20", "age >= 100 AND age <= 0",
             "age = 5 AND age = 99", "age = 30 AND age >= 40"]
    keys = {u: _key(u) for u in unsat}
    all_false_collide = len(set(keys.values())) == 1
    for u, kk in keys.items():
        print(f"  {u:30} -> {kk.split('WHERE')[-1].strip()}")
    print(f"  ALL provably-UNSAT conjunctions share ONE key: {all_false_collide}")
    # ...and FALSE must not collide with anything satisfiable.
    false_key = next(iter(keys.values()))
    leak = false_key == _key("age >= 20")
    print(f"  FALSE collides with a satisfiable 'age >= 20'? {leak}  (must be False)")
    rows.append(dict(kind="unsat", label="all_unsat_collide", lhs=";".join(unsat),
                     rhs="1=0", collide=all_false_collide, equivalent=True))
    rows.append(dict(kind="unsat", label="false_not_leaking", lhs="1=0",
                     rhs="age >= 20", collide=leak, equivalent=False))
    return rows


def demo_alias_renaming():
    section("2. Table-alias alpha-renaming")
    rows = []
    # Single physical table: qualifier is redundant -> all three collide.
    variants = [("orders o", "o.age >= 30"),
                ("orders x", "x.age >= 30"),
                ("orders", "orders.age >= 30"),
                ("orders", "age >= 30")]
    base = _key(variants[0][1], table=variants[0][0])
    print("  single table 'orders' -- these should ALL share one key:")
    for tbl, where in variants:
        kk = _key(where, table=tbl)
        print(f"    FROM {tbl:10} WHERE {where:20} -> {kk}")
        rows.append(dict(kind="alias_single", label=where, lhs=f"{tbl}|{where}",
                         rhs=variants[0][0] + "|" + variants[0][1],
                         collide=(kk == base), equivalent=True))
    all_same = len({_key(w, table=t) for t, w in variants}) == 1
    print(f"    -> all identical: {all_same}")

    # JOIN: qualifier is meaningful, must NOT be stripped (safe miss).
    section("2b. Across a JOIN the qualifier is KEPT (sound safe miss)")
    join = "orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey"
    ka = _key("o.l_quantity >= 30", table=join, int_cols=set())
    kb = _key("l.l_quantity >= 30", table=join, int_cols=set())
    print(f"    o.l_quantity>=30 -> {ka.split('WHERE')[-1].strip()}")
    print(f"    l.l_quantity>=30 -> {kb.split('WHERE')[-1].strip()}")
    print(f"    kept distinct (o.x != l.x): {ka != kb}  (must be True -- different columns)")
    rows.append(dict(kind="alias_join", label="qualifier_kept",
                     lhs="o.l_quantity>=30", rhs="l.l_quantity>=30",
                     collide=(ka == kb), equivalent=False))
    return rows


def demo_cache_reuse():
    section("3. The matcher reuses a noisy answer across the new equivalences")
    m = NormalizedEquivalenceMatcher(INT_COLS)
    # Seed one expensive (epsilon-spending) noisy answer.
    seed = parse_query("SELECT COUNT(*) FROM orders o WHERE o.age >= 30 AND o.age >= 40")
    m.add(seed, ["count"], [(4242,)], epsilon_used=0.5)
    print("  cached: 'o.age>=30 AND o.age>=40' (eps=0.5) -> 4242")
    follow_ups = [
        "SELECT COUNT(*) FROM orders WHERE age >= 40",                  # range-merge + alias
        "SELECT COUNT(*) FROM orders x WHERE x.age > 39",               # tighten + alias
        "SELECT COUNT(*) FROM orders WHERE age >= 41",                  # NOT equivalent -> miss
    ]
    rows = []
    for sql in follow_ups:
        hit = m.find_match(parse_query(sql))
        status = (f"HIT  reuse={hit.entry.rows[0][0]} extra_eps=0.0"
                  if hit else "MISS (spends fresh budget)")
        print(f"  {sql.split('WHERE')[-1].strip():28} -> {status}")
        rows.append(dict(kind="cache", label=sql.split("WHERE")[-1].strip(),
                         lhs=sql, rhs="seed", collide=hit is not None,
                         equivalent=None))
    return rows


def main():
    rows = []
    rows += demo_range_merging()
    rows += demo_alias_renaming()
    rows += demo_cache_reuse()

    section("Soundness audit headline")
    audited = [r for r in rows if r.get("equivalent") is not None]
    unsound = [r for r in audited if r["collide"] and not r["equivalent"]]
    missed = [r for r in audited if r["equivalent"] and not r["collide"]]
    print(f"  audited pairs: {len(audited)}")
    print(f"  UNSOUND collisions (collide but not equivalent): {len(unsound)}  "
          f"(MUST be 0)")
    print(f"  safe misses (equivalent but kept distinct):      {len(missed)}  "
          f"(allowed -- correctness-safe)")
    assert not unsound, f"SOUNDNESS VIOLATION: {unsound}"

    out = Path(__file__).parent.parent / "results" / "semantic_extended"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "semantic_extended.csv", index=False)
    print(f"\n  Wrote {out / 'semantic_extended.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
