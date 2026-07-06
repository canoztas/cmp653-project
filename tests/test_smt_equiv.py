"""SMT semantic-equivalence tests: power (catches what the normaliser cannot)
and SOUNDNESS (Z3 declares equivalence only when the predicates truly are).

Skipped entirely when z3 is not installed (it is an optional dependency)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.smt_equiv import (SMTEquivalenceMatcher, predicates_equivalent,
                            z3_available)
from dpdb.parser import parse_query

pytestmark = pytest.mark.skipif(not z3_available(), reason="z3 not installed")

INT = {"age", "education_num", "hours_per_week"}


def _where(sql_where, table="adult"):
    from sqlglot import exp
    q = parse_query(f"SELECT COUNT(*) FROM {table} WHERE {sql_where}")
    w = q.ast.find(exp.Where)
    return w.this if w is not None else None


def equiv(w1, w2, cols=INT):
    return predicates_equivalent(_where(w1), _where(w2), cols)


class TestSMTPower:
    """Equivalences beyond decidable predicate normalisation."""

    def test_linear_arithmetic(self):
        assert equiv("age*2 >= 60", "age >= 30") is True

    def test_disjunction_of_ranges(self):
        assert equiv("age >= 30 OR age >= 20", "age >= 20") is True

    def test_double_negation(self):
        assert equiv("NOT (age < 30)", "age >= 30") is True

    def test_de_morgan(self):
        assert equiv("NOT (age < 30 OR hours_per_week < 40)",
                     "age >= 30 AND hours_per_week >= 40") is True

    def test_cross_column(self):
        assert equiv("age > education_num", "education_num < age") is True


class TestSMTSoundness:
    def test_distinct_thresholds_not_equiv(self):
        assert equiv("age >= 30", "age >= 31") is False

    def test_different_categorical_not_equiv(self):
        assert equiv("sex = 'M'", "sex = 'F'") is False

    def test_arithmetic_non_equiv(self):
        # age*2 >= 61 is NOT age >= 30 (age=30 -> 60 < 61)
        assert equiv("age*2 >= 61", "age >= 30") is False

    def test_unsupported_returns_none(self):
        # a real-valued / unknown function is outside the sound fragment
        assert equiv("ABS(age) >= 30", "age >= 30") is None

    def test_bruteforce_no_false_collision(self):
        """Soundness property: whenever Z3 proves two integer predicates
        equivalent (over ALL integers), their truth tables over age in 0..120
        must coincide -- a false collision would be a translation bug. (The
        converse is NOT required: Z3 may conservatively say not-equivalent for
        predicates that happen to agree only on a bounded domain -- a safe miss.)
        We also exercise an equivalent-by-construction branch so the True path
        actually fires."""
        rng = random.Random(20260628)
        ops = ["<", "<=", ">", ">=", "=", "!="]
        dom = range(0, 121)
        checked = trues = 0

        def ev(atoms, joiner, a):
            vals = [fn(a) for _, fn in atoms]
            return all(vals) if joiner == " AND " else any(vals)

        for it in range(400):
            def atom():
                op = rng.choice(ops)
                c = rng.randint(-3, 124)
                mul = rng.choice([1, 1, 2])         # sometimes age*2
                lhs = f"age*{mul}" if mul != 1 else "age"
                return f"{lhs} {op} {c}", (lambda a, op=op, c=c, mul=mul: {
                    "<": a*mul < c, "<=": a*mul <= c, ">": a*mul > c,
                    ">=": a*mul >= c, "=": a*mul == c, "!=": a*mul != c}[op])
            a1 = [atom() for _ in range(rng.randint(1, 2))]
            j1 = rng.choice([" AND ", " OR "])
            s1 = j1.join(x[0] for x in a1)
            if it % 3 == 0:                          # construct a true equivalence
                s2, a2, j2 = s1, a1, j1               # syntactically identical -> equivalent
                # perturb with a tightening rewrite that preserves meaning on ints
                s2 = s1.replace("age >", "age*1 >")  # harmless arithmetic identity
            else:
                a2 = [atom() for _ in range(rng.randint(1, 2))]
                j2 = rng.choice([" AND ", " OR "])
                s2 = j2.join(x[0] for x in a2)
            verdict = equiv(s1, s2)
            if verdict is None:
                continue
            checked += 1
            same = all(ev(a1, j1, a) == ev(a2, j2, a) for a in dom)
            if verdict is True:
                trues += 1
                assert same, f"UNSOUND false collision: [{s1}] vs [{s2}]"
        assert checked > 50 and trues > 0   # solver exercised, True path fired


class TestSMTMatcher:
    def test_matcher_reuses_arithmetic_equivalent(self):
        m = SMTEquivalenceMatcher(INT)
        m.add(parse_query("SELECT COUNT(*) FROM adult WHERE age >= 30"),
              ["c"], [(100,)], 1.0)
        hit = m.find_match(parse_query("SELECT COUNT(*) FROM adult WHERE age*2 >= 60"))
        assert hit is not None and hit.entry.rows == [(100,)]
        miss = m.find_match(parse_query("SELECT COUNT(*) FROM adult WHERE age >= 31"))
        assert miss is None

    def test_matcher_respects_structure(self):
        # same WHERE but a different aggregate must NOT match (different bucket).
        m = SMTEquivalenceMatcher(INT)
        m.add(parse_query("SELECT COUNT(*) FROM adult WHERE age >= 30"),
              ["c"], [(100,)], 1.0)
        other = m.find_match(parse_query("SELECT SUM(age) FROM adult WHERE age >= 30"))
        assert other is None
