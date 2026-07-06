"""Equivalence-aware forecasting: the forecast prices distinct equivalence
CLASSES, not byte-identical query strings.

Each logical query is issued in one of several syntactically different but
provably equivalent forms (e.g. the class "age>=30" appears as "age >= 30",
"age > 29", "30 <= age", or "age >= 20 AND age >= 30"). A byte-identical cache
treats these as distinct species and over-counts u_k; the equivalence-aware
cache (predicate normalisation, dpdb.normalize) collapses them to one class.

We draw a Zipf workload over the logical classes, emit a random equivalent form
each time, and check that (i) the realised distinct-CLASS count matches the
occupancy forecast E[u_k] over the class distribution, and (ii) a byte-identical
cache over-counts (and so over-spends) by the variant multiplicity. This is what
makes the forecast equivalence-aware rather than tied to exact repeats.

Deterministic seeds. Run: python experiments/equivalence_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.normalize import canonical_sql
from dpdb.parser import parse_query
from dpdb.model import expected_unique_queries, zipf_distribution

INT = {"age"}
SEXES = ("'M'", "'F'")
ALPHAS, KS, TRIALS = (0.5, 1.0, 1.5), (50, 150), 40


def logical_classes(n=10):
    """n logical predicates, each with several provably-equivalent SQL forms."""
    classes = []
    for j in range(n):
        c = 20 + 5 * j                         # threshold
        sex = SEXES[j % 2] if j % 3 else None  # some classes add a categorical
        base = f"age >= {c}"
        variants = [base, f"age > {c-1}", f"{c} <= age", f"age >= {c-10} AND age >= {c}"]
        if sex is not None:
            variants = [f"{v} AND sex = {sex}" for v in variants]
            # also a reordered form
            variants.append(f"sex = {sex} AND age > {c-1}")
        classes.append(variants)
    return classes


def canon(where):
    return canonical_sql(parse_query(f"SELECT COUNT(*) FROM adult WHERE {where}"), INT)


def main():
    classes = logical_classes(10)
    L = len(classes)
    # sanity: every variant of a class shares ONE canonical key, distinct across classes
    keys = [canon(v) for v in (cl[0] for cl in classes)]
    assert len(set(keys)) == L, "logical classes must be pairwise distinct"
    for cl in classes:
        assert len({canon(v) for v in cl}) == 1, "variants of a class must collapse"

    print("=== Equivalence-aware forecast: realized distinct CLASSES vs E[u_k] ===")
    print(f"    {L} logical classes, each with {len(classes[0])} equivalent SQL forms\n")
    print(f"  {'alpha':>5} {'k':>4} | {'E[u_k] (class)':>13} {'classes (eq cache)':>18} "
          f"{'strings (byte cache)':>20} {'over-count':>10}")
    rows = []
    for alpha in ALPHAS:
        p = zipf_distribution(L, alpha)
        for k in KS:
            euk = expected_unique_queries(p, k)
            eq_u, str_u = [], []
            for t in range(TRIALS):
                rng = np.random.default_rng(7000 + 13 * k + int(alpha * 100) + t)
                idx = rng.choice(L, size=k, p=p)
                eq_keys, str_keys = set(), set()
                for j in idx:
                    v = classes[j][rng.integers(len(classes[j]))]   # random equiv form
                    eq_keys.add(canon(v))                            # equivalence class
                    str_keys.add(v)                                 # byte-identical
                eq_u.append(len(eq_keys)); str_u.append(len(str_keys))
            eu, su = float(np.mean(eq_u)), float(np.mean(str_u))
            rows.append(dict(alpha=alpha, k=k, Euk=euk, classes=eu, strings=su,
                             rel_err_pct=abs(eu - euk) / euk * 100, over=su / eu))
            print(f"  {alpha:>5.1f} {k:>4} | {euk:13.3f} {eu:18.2f} {su:20.2f} "
                  f"{su/eu:9.2f}x")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "equivalence"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "equivalence_forecast.csv", index=False)

    print("\n=== Headline ===")
    print(f"  Realized distinct equivalence classes match the occupancy forecast "
          f"E[u_k] over the\n    class distribution to a mean {df.rel_err_pct.mean():.2f}% "
          f"(max {df.rel_err_pct.max():.2f}%).")
    print(f"  A byte-identical cache over-counts the species by "
          f"{df.over.min():.2f}-{df.over.max():.2f}x (the variant multiplicity), "
          f"and would over-spend by the same factor.")
    print(f"  => the equivalence-aware forecast prices semantically-distinct "
          f"queries, not exact strings.")
    print(f"\n  Wrote {out / 'equivalence_forecast.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
