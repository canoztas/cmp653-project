"""Sound predicate normalisation for an equivalence-checked cache.

The exact-repeat cache keys on structural identity, so it misses textbook
equivalences (``age>=30`` vs ``age>29``, AND/OR reordering). A *similarity*
score (tree kernel / embedding, see :mod:`dpdb.semantic`) is unsound for a cache:
it can admit a non-equivalent query and return the wrong answer. This module
takes the safe route -- a symbolic *normal form* under which two queries collide
ONLY when they are provably equivalent, so the cache never reuses a wrong answer
(no false positives), and a reused noisy answer is still post-processing (zero
extra budget).

Decidable, exact equivalences captured:
  * integer-comparison tightening:  ``x > c`` == ``x >= c+1`` and
    ``x < c`` == ``x <= c-1`` -- sound ONLY for integer-typed columns, so it is
    applied solely to columns the caller declares integral;
  * column-on-left orientation:     ``c < x`` == ``x > c`` (and friends);
  * commutative reordering:         ``a AND b`` == ``b AND a`` (likewise OR);
  * range-merging on one integer column within a conjunction: redundant bounds
    collapse to the tightest one (``age>=30 AND age>=40`` == ``age>=40``;
    ``age<=50 AND age<=70`` == ``age<=50``), and a contradictory conjunction
    (``age>=40 AND age<=20``, or an equality outside its surviving bounds, or
    two distinct equalities on one column) canonicalises to a single FALSE so
    that ALL provably-unsatisfiable conjunctions collide with one another;
  * table-alias alpha-renaming: when the query references exactly ONE physical
    table, a column's table qualifier is redundant, so ``a.age``, ``orders.age``
    and bare ``age`` all canonicalise identically. With two or more tables (a
    join) the qualifier is meaningful and is KEPT -- stripping it could collide
    two different physical columns, so that case is a safe miss.

Anything not provably equivalent is left distinct -- a correctness-safe miss,
never a wrong reuse. Richer equivalences (cross-column arithmetic, joins, the
full theory of linear integer arithmetic) need an SMT equivalence check and are
out of scope: when range-merging cannot decide a conjunction soundly it leaves
the predicates untouched rather than guess. This module is the small, sound
first step toward the equivalence-checked cache discussed in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from sqlglot import exp

from dpdb.parser import ParsedQuery

# Comparison operator that results from moving the column to the left-hand side.
_FLIP = {
    exp.LT: exp.GT, exp.GT: exp.LT,
    exp.LTE: exp.GTE, exp.GTE: exp.LTE,
    exp.EQ: exp.EQ, exp.NEQ: exp.NEQ,
}
_CMP = tuple(_FLIP)

# A conjunction proven unsatisfiable canonicalises to this single literal, so
# every provably-FALSE WHERE collides with every other (and with nothing that
# can ever be true). ``1 = 0`` is a closed integer literal, dialect-stable.
_FALSE = exp.EQ(this=exp.Literal.number(1), expression=exp.Literal.number(0))


def _normalize_cmp(node: exp.Expression, int_cols: frozenset) -> exp.Expression:
    """Orient the column left and tighten integer ``>``/``<`` to ``>=``/``<=``."""
    if not isinstance(node, _CMP):
        return node
    left, right = node.this, node.expression
    if isinstance(left, exp.Literal) and isinstance(right, exp.Column):
        node = _FLIP[type(node)](this=right.copy(), expression=left.copy())
        left, right = node.this, node.expression
    if (isinstance(left, exp.Column) and isinstance(right, exp.Literal)
            and right.is_int and left.name in int_cols):
        c = int(right.this)
        if isinstance(node, exp.GT):        # x > c  <=>  x >= c+1   (integers)
            return exp.GTE(this=left.copy(), expression=exp.Literal.number(c + 1))
        if isinstance(node, exp.LT):        # x < c  <=>  x <= c-1   (integers)
            return exp.LTE(this=left.copy(), expression=exp.Literal.number(c - 1))
    return node


def _flatten(node: exp.Expression, typ) -> list:
    if isinstance(node, typ):
        return _flatten(node.left, typ) + _flatten(node.right, typ)
    return [node]


def _col_key(col: exp.Column) -> str:
    """Identity of a column for grouping. After alias stripping the qualifier is
    gone, so this is just the bare name; when a qualifier survives (multi-table)
    it is part of the key so ``a.age`` and ``b.age`` are never merged together."""
    return col.sql(dialect="postgres")


def _int_literal_cmp(node: exp.Expression, int_cols: frozenset):
    """If ``node`` is ``<int-col> <op> <int-literal>`` return (key, op, value),
    else None. ``node`` is assumed already cmp-normalised (column on the left,
    GT/LT already tightened to GTE/LTE), so ``op`` is one of >=, <=, =, !=."""
    if not isinstance(node, (exp.GTE, exp.LTE, exp.EQ, exp.NEQ)):
        return None
    left, right = node.this, node.expression
    if not (isinstance(left, exp.Column) and isinstance(right, exp.Literal)):
        return None
    if not (right.is_int and left.name in int_cols):
        return None
    return _col_key(left), type(node), int(right.this)


def _merge_one_column(col_sql: str, items: list) -> Optional[list]:
    """Merge all integer literal comparisons on a SINGLE column into the tightest
    equivalent set, or signal unsatisfiability.

    ``items`` is a list of (op_type, value). Returns either:
      * ``None``                      -> the conjunction is UNSAT (return FALSE),
      * a list of replacement nodes   -> the canonical, redundancy-free bounds.

    The transformation is a pure logical equivalence over any totally ordered
    integer domain: a conjunction of ``>=`` bounds equals the single greatest
    lower bound; of ``<=`` bounds the single least upper bound; an equality pins
    a point that must lie inside the surviving window and agree with every other
    equality, and a disequality only matters when it excludes that point."""
    lowers = [v for t, v in items if t is exp.GTE]   # x >= v
    uppers = [v for t, v in items if t is exp.LTE]   # x <= v
    eqs = {v for t, v in items if t is exp.EQ}       # x = v
    neqs = {v for t, v in items if t is exp.NEQ}     # x != v

    lo = max(lowers) if lowers else None             # tightest lower bound
    hi = min(uppers) if uppers else None             # tightest upper bound

    if lo is not None and hi is not None and lo > hi:
        return None                                  # x>=lo AND x<=hi, lo>hi -> FALSE
    if len(eqs) > 1:
        return None                                  # x=a AND x=b, a!=b -> FALSE
    col = exp.column(col_sql.split(".")[-1],
                     table=col_sql.split(".")[0] if "." in col_sql else None)

    if eqs:
        (point,) = tuple(eqs)
        if lo is not None and point < lo:
            return None                              # x=point AND x>=lo, point<lo
        if hi is not None and point > hi:
            return None                              # x=point AND x<=hi, point>hi
        if point in neqs:
            return None                              # x=point AND x!=point -> FALSE
        # x=point subsumes every bound and every other disequality (they cannot
        # contradict it without an earlier UNSAT). Canonical form is just x=point.
        return [exp.EQ(this=col.copy(), expression=exp.Literal.number(point))]

    out: list = []
    if lo is not None:
        out.append(exp.GTE(this=col.copy(), expression=exp.Literal.number(lo)))
    if hi is not None:
        out.append(exp.LTE(this=col.copy(), expression=exp.Literal.number(hi)))
    # Disequalities are kept verbatim, but a "!= v" with v strictly outside the
    # surviving [lo, hi] window is vacuously true and is dropped (sound: it
    # removes nothing from the satisfying set). Anything else stays untouched.
    for v in sorted(neqs):
        outside_low = lo is not None and v < lo
        outside_high = hi is not None and v > hi
        if outside_low or outside_high:
            continue
        out.append(exp.NEQ(this=col.copy(), expression=exp.Literal.number(v)))
    return out


def _merge_ranges(node: exp.Expression, int_cols: frozenset) -> exp.Expression:
    """Range-merge integer literal comparisons sharing a column inside an AND."""
    if not isinstance(node, exp.And):
        return node
    parts = _flatten(node, exp.And)

    grouped: dict[str, list] = {}
    others: list = []
    order: list[str] = []
    for p in parts:
        info = _int_literal_cmp(p, int_cols)
        if info is None:
            others.append(p)
            continue
        key, op, val = info
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((op, val))

    merged: list = []
    for key in order:
        replacement = _merge_one_column(key, grouped[key])
        if replacement is None:
            return _FALSE.copy()                     # whole conjunction is FALSE
        merged.extend(replacement)

    new_parts = merged + others
    if not new_parts:                                # defensive: nothing to AND
        return node
    acc = new_parts[0]
    for p in new_parts[1:]:
        acc = exp.And(this=acc, expression=p)
    return acc


def _sort_connectors(node: exp.Expression) -> exp.Expression:
    """Rebuild AND/OR chains in a canonical (sorted) operand order."""
    if not isinstance(node, (exp.And, exp.Or)):
        return node
    typ = type(node)
    parts = sorted(_flatten(node, typ), key=lambda n: n.sql(dialect="postgres"))
    acc = parts[0]
    for p in parts[1:]:
        acc = typ(this=acc, expression=p)
    return acc


def _single_physical_table(parsed: ParsedQuery) -> bool:
    """True iff the query reads exactly one physical table. Only then is a column
    qualifier redundant and safe to strip (a join makes ``a.x`` vs ``b.x``
    distinguish two different physical columns, so we must NOT strip there)."""
    tables = {t for t in (parsed.tables or []) if t}
    if not tables:
        # Fall back to scanning the AST for table names if the parser did not
        # populate ``tables`` (be conservative: treat unknown as multi-table).
        names = {t.name for t in parsed.ast.find_all(exp.Table)}
        return len(names) == 1
    return len(tables) == 1


def _strip_alias(node: exp.Expression) -> exp.Expression:
    """Drop table aliases (alpha-renaming) so the qualifier carries no identity:
    a column ``a.age`` / ``orders.age`` becomes bare ``age``, and the FROM table
    ``orders AS o`` loses its ``AS o``. Sound only when the caller has
    established there is a single physical table; guarded by the caller."""
    if isinstance(node, exp.Column) and node.table:
        return exp.column(node.name)
    if isinstance(node, exp.Table) and node.alias:
        bare = node.copy()
        bare.set("alias", None)
        return bare
    return node


def canonical_sql(parsed: ParsedQuery, int_columns: Iterable[str]) -> str:
    """Canonical normal-form SQL: equal iff provably equivalent (given the
    integer-typed columns named in ``int_columns``)."""
    int_cols = frozenset(int_columns)
    tree = parsed.ast
    # Alias alpha-renaming FIRST: only when a single physical table makes the
    # qualifier redundant, so equal columns group identically for range-merging.
    if _single_physical_table(parsed):
        tree = tree.transform(_strip_alias)
    tree = tree.transform(lambda n: _normalize_cmp(n, int_cols))
    tree = tree.transform(lambda n: _merge_ranges(n, int_cols))
    tree = tree.transform(_sort_connectors)
    return tree.sql(dialect="postgres")


# --- A sound drop-in for the SEMANTIC_AWARE cache layer -----------------------
# Same minimal interface the ledger expects of a matcher (add / find_match),
# so it plugs into BudgetLedger(strategy=SEMANTIC_AWARE, semantic_matcher=...)
# with no change to the budget code -- but it admits a match ONLY on provable
# equivalence, unlike the similarity matcher in dpdb.semantic.

@dataclass
class _Entry:
    columns: list
    rows: list
    epsilon_used: float


@dataclass
class _Match:
    entry: _Entry


class NormalizedEquivalenceMatcher:
    """Exact equivalence cache keyed on the predicate normal form."""

    def __init__(self, int_columns: Iterable[str] = ()):
        self.int_columns = frozenset(int_columns)
        self._by_key: dict[str, _Entry] = {}

    def _key(self, parsed: ParsedQuery) -> str:
        return canonical_sql(parsed, self.int_columns)

    def add(self, parsed: ParsedQuery, columns, rows, epsilon_used: float) -> None:
        self._by_key[self._key(parsed)] = _Entry(columns, rows, epsilon_used)

    def find_match(self, parsed: ParsedQuery) -> Optional[_Match]:
        entry = self._by_key.get(self._key(parsed))
        return _Match(entry) if entry is not None else None
