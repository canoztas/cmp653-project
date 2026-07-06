"""Soundness + coverage tests for the equivalence-checked cache normaliser.

A cache may collide two queries ONLY when they are provably equivalent. These
tests pin both directions: the textbook equivalences the exact cache misses
DO collide, and non-equivalent queries do NOT (no false positives, hence no
wrong-answer reuse).

The final class is PROPERTY-BASED: it generates random predicate pairs over a
small integer domain (age in 0..120), and asserts the cache's golden rule by
brute force -- every pair the normaliser COLLIDES is genuinely equivalent
(identical satisfying sets), with NO false positives ever; and a battery of
known-equivalent pairs DO collide so the merging actually fires.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.normalize import NormalizedEquivalenceMatcher, canonical_sql
from dpdb.parser import parse_query

INT = {"age"}


def _q(where, table="adult"):
    return parse_query(f"SELECT COUNT(*) FROM {table} WHERE {where}")


def key(where, int_cols=INT, table="adult"):
    return canonical_sql(_q(where, table), int_cols)


class TestProvableEquivalence:
    def test_integer_tightening_collides(self):
        # age > 29  ==  age >= 30   (integers)
        assert key("age > 29") == key("age >= 30")

    def test_strict_less_tightening(self):
        # age < 30  ==  age <= 29
        assert key("age < 30") == key("age <= 29")

    def test_and_commutativity_collides(self):
        assert key("age >= 30 AND sex = 'M'") == key("sex = 'M' AND age >= 30")

    def test_column_orientation_collides(self):
        # 30 <= age  ==  age >= 30
        assert key("30 <= age") == key("age >= 30")

    def test_combined_equivalence(self):
        assert key("age > 29 AND sex = 'M'") == key("sex = 'M' AND age >= 30")


class TestSoundness:
    def test_non_equivalent_thresholds_distinct(self):
        # age >= 30 is NOT age >= 31
        assert key("age >= 30") != key("age >= 31")

    def test_different_literal_distinct(self):
        assert key("sex = 'M'") != key("sex = 'F'")

    def test_tightening_not_applied_to_non_integer_columns(self):
        # If the column is not declared integral, > and >= must NOT be merged:
        # for reals, age > 29 is not age >= 30. Safe miss, never a false match.
        assert key("age > 29", int_cols=set()) != key("age >= 30", int_cols=set())


class TestRangeMerging:
    def test_redundant_lower_bound_collapses(self):
        # age>=30 AND age>=40  ==  age>=40
        assert key("age >= 30 AND age >= 40") == key("age >= 40")

    def test_redundant_upper_bound_collapses(self):
        # age<=70 AND age<=50  ==  age<=50
        assert key("age <= 70 AND age <= 50") == key("age <= 50")

    def test_redundant_bound_with_tightening(self):
        # age>29 AND age>39  ==  age>=40 (tighten then merge)
        assert key("age > 29 AND age > 39") == key("age >= 40")

    def test_unsatisfiable_lower_above_upper_is_false(self):
        # age>=40 AND age<=20  is UNSAT -> canonical FALSE
        unsat = key("age >= 40 AND age <= 20")
        other_unsat = key("age >= 100 AND age <= 0")
        assert unsat == other_unsat                 # all FALSE conjunctions collide
        # ...and an UNSAT predicate must NOT collide with any satisfiable one.
        assert unsat != key("age >= 20")
        assert unsat != key("age <= 40")

    def test_two_distinct_equalities_are_false(self):
        assert key("age = 30 AND age = 40") == key("age >= 100 AND age <= 0")

    def test_equality_outside_bound_is_false(self):
        # age=30 AND age>=40 is UNSAT
        assert key("age = 30 AND age >= 40") == key("age >= 100 AND age <= 0")

    def test_equality_inside_bound_subsumes(self):
        # age=30 AND age>=20  ==  age=30
        assert key("age = 30 AND age >= 20") == key("age = 30")

    def test_merging_keeps_other_columns(self):
        # Range merge on age must leave the sex predicate intact, not drop it.
        assert key("age >= 30 AND age >= 40 AND sex = 'M'") == \
            key("age >= 40 AND sex = 'M'")
        assert key("age >= 30 AND age >= 40 AND sex = 'M'") != key("age >= 40")

    def test_non_redundant_bounds_distinct(self):
        # A genuine window must NOT collapse to one bound (soundness).
        assert key("age >= 30 AND age <= 50") != key("age >= 30")
        assert key("age >= 30 AND age <= 50") != key("age <= 50")

    def test_real_column_not_merged(self):
        # Without the integer declaration we do NOT range-merge strict bounds:
        # range-merging only reasons about the canonical >=/<=/=/!= forms that
        # integer tightening produces. For a non-integer column `age > 30 AND
        # age > 40` is left as two predicates (a safe miss), never collapsed.
        assert key("age > 30 AND age > 40", int_cols=set()) != \
            key("age > 40", int_cols=set())
        # Non-strict bounds ARE sound to merge on any ordered domain, and they
        # are emitted by tightening, so >= still collapses even without the int
        # declaration's strict->non-strict step... but here, with no int decl,
        # the `>` never becomes `>=`, so nothing merges: distinct, and safe.


class TestAliasRenaming:
    def test_qualifier_stripped_single_table(self):
        # orders.age, o.age and bare age all canonicalise identically.
        assert key("o.age >= 30", table="orders o") == key("age >= 30", table="orders")

    def test_alias_renaming_is_alpha_invariant(self):
        # The specific alias letter must not matter.
        assert key("o.age >= 30", table="orders o") == \
            key("x.age >= 30", table="orders x")

    def test_alias_strip_enables_range_merge(self):
        # o.age and bare age are the SAME column, so the redundant bound merges.
        assert key("o.age >= 30 AND age >= 40", table="orders o") == \
            key("age >= 40", table="orders")

    def test_join_keeps_qualifiers_distinct(self):
        # Two physical tables: o.x and l.x are DIFFERENT columns, qualifier kept.
        join = ("orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey")
        a = key("o.l_quantity >= 30", int_cols=set(), table=join)
        b = key("l.l_quantity >= 30", int_cols=set(), table=join)
        assert a != b                                # safe miss, never a false match


class TestMatcher:
    def test_matcher_reuses_equivalent_and_rejects_non_equivalent(self):
        m = NormalizedEquivalenceMatcher(INT)
        m.add(_q("age > 29"), ["count"], [(100,)], 1.0)
        # equivalent query -> hit, same cached answer, zero extra budget
        hit = m.find_match(_q("age >= 30"))
        assert hit is not None and hit.entry.rows == [(100,)] and hit.entry.epsilon_used == 1.0
        # non-equivalent query -> miss
        assert m.find_match(_q("age >= 31")) is None


# --------------------------------------------------------------------------- #
# Property-based: the cache's golden rule, checked by brute force.
#
# We generate random conjunctions of integer comparisons on `age` (the feature
# range-merging actually reasons about) plus an optional categorical predicate,
# render each to SQL, and ALSO evaluate it directly over age in 0..120. The
# normaliser may collide two predicates ONLY if their satisfying sets are equal.
# --------------------------------------------------------------------------- #

DOMAIN = range(0, 121)          # age in 0..120 (inclusive), the brute-force world
SEXES = ("'M'", "'F'")
_OPS = {
    ">":  lambda a, c: a > c,
    ">=": lambda a, c: a >= c,
    "<":  lambda a, c: a < c,
    "<=": lambda a, c: a <= c,
    "=":  lambda a, c: a == c,
    "!=": lambda a, c: a != c,
}


def _random_atom(rng):
    """A single integer comparison on age, returned as (sql, predicate(age))."""
    op = rng.choice(list(_OPS))
    c = rng.randint(-5, 125)            # spill just outside the domain on purpose
    fn = _OPS[op]
    return f"age {op} {c}", (lambda a, fn=fn, c=c: fn(a, c))


def _random_predicate(rng):
    """A conjunction of 1..3 age atoms, optionally AND a sex equality.

    Returns (sql_string, satisfies(age, sex)). The brute-force oracle evaluates
    the FULL predicate -- age atoms and the optional categorical -- over the
    joint domain, so the equivalence check is exact regardless of which side
    carries a sex clause."""
    n = rng.randint(1, 3)
    atoms = [_random_atom(rng) for _ in range(n)]
    sql_atoms = [a[0] for a in atoms]
    fns = [a[1] for a in atoms]
    sex_val = None
    if rng.random() < 0.4:
        sex_val = rng.choice(SEXES).strip("'")
        sql_atoms.append(f"sex = '{sex_val}'")
    sql = " AND ".join(sql_atoms)

    def satisfies(age, sex):
        if not all(fn(age) for fn in fns):
            return False
        if sex_val is not None and sex != sex_val:
            return False
        return True

    return sql, satisfies


# Joint brute-force domain: age x sex. Two predicates are equivalent iff they
# accept exactly the same (age, sex) pairs.
_JOINT = [(a, s) for a in DOMAIN for s in ("M", "F")]


def _sat_set(satisfies):
    return frozenset((a, s) for a, s in _JOINT if satisfies(a, s))


class TestPropertyBasedSoundness:
    def test_collisions_are_genuinely_equivalent(self):
        """No false positives: any colliding pair has identical satisfying sets.

        We fix the categorical predicate to the SAME value on both sides so the
        only thing that can make satisfying sets differ is the age logic, which
        the normaliser is supposed to reason about exactly."""
        rng = random.Random(20260627)
        n_pairs = 4000
        collisions = 0
        for _ in range(n_pairs):
            sql_a, sat_a = _random_predicate(rng)
            sql_b, sat_b = _random_predicate(rng)
            ka, kb = key(sql_a), key(sql_b)
            if ka == kb:
                collisions += 1
                set_a = _sat_set(sat_a)
                set_b = _sat_set(sat_b)
                # Keys collide -> the predicates MUST accept identical (age, sex)
                # pairs over the whole domain. Any mismatch is an unsound reuse.
                assert set_a == set_b, (
                    f"FALSE POSITIVE: {sql_a!r} collided with {sql_b!r} "
                    f"but sat sets differ ({sorted(set_a)} vs {sorted(set_b)})")
        # Sanity: the random stream actually produced collisions to check.
        assert collisions > 50, f"too few collisions to be meaningful: {collisions}"

    def test_known_equivalent_pairs_collide(self):
        """Coverage: equivalences the normaliser is meant to capture DO fire."""
        equiv_pairs = [
            ("age > 29", "age >= 30"),
            ("age < 41", "age <= 40"),
            ("age >= 30 AND age >= 40", "age >= 40"),
            ("age <= 60 AND age <= 80", "age <= 60"),
            ("age > 29 AND age > 39 AND age > 9", "age >= 40"),
            ("age >= 40 AND age <= 20", "age = 5 AND age = 99"),  # both UNSAT
            ("age = 30 AND age >= 10", "age = 30"),
            ("50 >= age", "age <= 50"),
        ]
        for a, b in equiv_pairs:
            assert _sat_set(_pred_fn(a)) == _sat_set(_pred_fn(b)), \
                f"test bug: {a!r} and {b!r} are not actually equivalent"
            assert key(a) == key(b), f"missed equivalence: {a!r} vs {b!r}"

    def test_known_distinct_pairs_stay_distinct(self):
        """The flip side: genuinely different predicates must NOT collide."""
        distinct_pairs = [
            ("age >= 30", "age >= 31"),
            ("age >= 30 AND age <= 50", "age >= 30"),
            ("age > 29", "age > 30"),
            ("age = 30", "age = 31"),
            ("age >= 40 AND age <= 20", "age >= 20"),   # UNSAT vs satisfiable
        ]
        for a, b in distinct_pairs:
            assert _sat_set(_pred_fn(a)) != _sat_set(_pred_fn(b)), \
                f"test bug: {a!r} and {b!r} ARE equivalent"
            assert key(a) != key(b), f"unsound collision: {a!r} vs {b!r}"


def _pred_fn(sql):
    """Parse a simple AND-of-age-comparisons SQL string into a Python predicate
    over (age, sex), for the brute-force oracle. Supports only the grammar the
    tests above use (age <op> int / int <op> age, joined by AND). The returned
    predicate ignores sex (none of these literals constrain it)."""
    fns = []
    for part in sql.split(" AND "):
        part = part.strip()
        for op in ("<=", ">=", "!=", "=", "<", ">"):
            if op in part:
                lhs, rhs = part.split(op, 1)
                lhs, rhs = lhs.strip(), rhs.strip()
                if lhs == "age":
                    c = int(rhs)
                    fns.append((lambda a, op=op, c=c: _OPS[op](a, c)))
                elif rhs == "age":
                    c = int(lhs)
                    flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<=",
                            "=": "=", "!=": "!="}[op]
                    fns.append((lambda a, flip=flip, c=c: _OPS[flip](a, c)))
                break
    return lambda a, s: all(fn(a) for fn in fns)
