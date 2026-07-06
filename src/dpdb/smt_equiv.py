"""SMT-backed semantic equivalence for the cache (optional, sound).

The decidable predicate normaliser (:mod:`dpdb.normalize`) collides queries only
when a small set of syntactic rules proves them equivalent. A *symbolic solver*
(SMT) decides a much larger fragment: cross-column comparisons, arithmetic on
integer columns (``age*2>=60`` == ``age>=30``), disjunctions of ranges,
De Morgan, and double negation. This module asks Z3 whether two queries'
predicates are logically equivalent and, if so, lets the cache reuse the noisy
answer (still post-processing, zero extra budget).

SOUNDNESS is absolute: a match is admitted ONLY when Z3 *proves* the predicates
equivalent (``Not(p1 == p2)`` is UNSAT), so there are no false collisions. To
stay sound we model:
  * integer-typed columns (declared by the caller) as unbounded ``z3.Int`` --
    so an equivalence is admitted only if it holds for ALL integers, a subset of
    the true equivalences and never a superset;
  * categorical columns as integers restricted to ``=`` / ``!=`` against string
    literals (equality is domain-agnostic); inequalities/arithmetic on a
    non-integer column are UNSUPPORTED and yield no match (a safe miss).
Anything the translator does not fully understand raises ``_Unsupported`` and the
pair is left distinct. Z3 is an OPTIONAL dependency: if it is not installed the
matcher degrades to "no semantic match" (it never guesses).

Two queries are compared only when their non-WHERE structure (table, aggregates,
GROUP BY, ORDER/LIMIT) is identical; only the WHERE predicate is solved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from sqlglot import exp

from dpdb.parser import ParsedQuery

try:                                  # optional dependency
    import z3
except Exception:                     # pragma: no cover - environment without z3
    z3 = None


class _Unsupported(Exception):
    """Raised when a node is outside the sound translatable fragment."""


def z3_available() -> bool:
    return z3 is not None


def _structural_sig(parsed: ParsedQuery) -> tuple:
    """Hashable signature of everything EXCEPT the WHERE predicate. Two queries
    with the same signature compute the same thing up to their filter."""
    aggs = tuple(sorted((a.func, a.column, a.position) for a in parsed.aggregates))
    return (parsed.table, tuple(parsed.tables), aggs, tuple(parsed.group_by),
            parsed.order_by_position, parsed.order_desc, parsed.limit,
            parsed.is_join, parsed.join_table, parsed.join_on)


def _term(node: exp.Expression, env: dict, intcols: frozenset):
    """Translate an arithmetic term over integer columns to a z3 Int term."""
    if isinstance(node, exp.Paren):
        return _term(node.this, env, intcols)
    if isinstance(node, exp.Column):
        if node.name not in intcols:
            raise _Unsupported(f"arithmetic on non-integer column {node.name}")
        return env.setdefault(node.name, z3.Int(node.name))
    if isinstance(node, exp.Literal):
        if node.is_int:
            return z3.IntVal(int(node.this))
        raise _Unsupported("non-integer literal in arithmetic")
    if isinstance(node, exp.Neg):
        return -_term(node.this, env, intcols)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        a, b = _term(node.this, env, intcols), _term(node.expression, env, intcols)
        return a + b if isinstance(node, exp.Add) else (a - b if isinstance(node, exp.Sub) else a * b)
    raise _Unsupported(f"unsupported arithmetic node {type(node).__name__}")


def _is_str_literal(n) -> bool:
    return isinstance(n, exp.Literal) and n.is_string


def _pred(node: exp.Expression, env: dict, strmap: dict, intcols: frozenset):
    """Translate a boolean predicate to a z3 BoolRef (sound fragment only)."""
    if isinstance(node, exp.Paren):
        return _pred(node.this, env, strmap, intcols)
    if isinstance(node, exp.And):
        return z3.And(_pred(node.this, env, strmap, intcols),
                      _pred(node.expression, env, strmap, intcols))
    if isinstance(node, exp.Or):
        return z3.Or(_pred(node.this, env, strmap, intcols),
                     _pred(node.expression, env, strmap, intcols))
    if isinstance(node, exp.Not):
        return z3.Not(_pred(node.this, env, strmap, intcols))
    if isinstance(node, (exp.EQ, exp.NEQ)):
        left, right = node.this, node.expression
        # categorical equality: column <op> string-literal (domain-agnostic)
        col, lit = None, None
        if isinstance(left, exp.Column) and _is_str_literal(right):
            col, lit = left, right
        elif isinstance(right, exp.Column) and _is_str_literal(left):
            col, lit = right, left
        if col is not None:
            var = env.setdefault(col.name, z3.Int(col.name))
            const = strmap.setdefault(str(lit.this), len(strmap) + 1)  # distinct int per literal
            eq = var == z3.IntVal(const)
            return eq if isinstance(node, exp.EQ) else z3.Not(eq)
        # otherwise numeric (integer) equality
        a, b = _term(left, env, intcols), _term(right, env, intcols)
        return (a == b) if isinstance(node, exp.EQ) else (a != b)
    if isinstance(node, (exp.LT, exp.LTE, exp.GT, exp.GTE)):
        a, b = _term(node.this, env, intcols), _term(node.expression, env, intcols)
        if isinstance(node, exp.LT):
            return a < b
        if isinstance(node, exp.LTE):
            return a <= b
        if isinstance(node, exp.GT):
            return a > b
        return a >= b
    raise _Unsupported(f"unsupported predicate node {type(node).__name__}")


def predicates_equivalent(where1: Optional[exp.Expression],
                          where2: Optional[exp.Expression],
                          int_columns: Iterable[str]) -> Optional[bool]:
    """Return True/False if Z3 decides the two WHERE predicates equivalent, or
    None if Z3 is unavailable, the fragment is unsupported, or the solver returns
    unknown. A None result must be treated as 'no match' by callers."""
    if z3 is None:
        return None
    intcols = frozenset(int_columns)
    env: dict = {}
    strmap: dict = {}
    try:
        p1 = z3.BoolVal(True) if where1 is None else _pred(where1, env, strmap, intcols)
        p2 = z3.BoolVal(True) if where2 is None else _pred(where2, env, strmap, intcols)
    except _Unsupported:
        return None
    solver = z3.Solver()
    solver.set("timeout", 2000)            # ms; unknown -> no match
    solver.add(z3.Not(p1 == p2))
    r = solver.check()
    if r == z3.unsat:
        return True                        # proven equivalent
    if r == z3.sat:
        return False
    return None                            # unknown / timeout


# --- A sound drop-in for the SEMANTIC_AWARE cache layer ----------------------

@dataclass
class _Entry:
    columns: list
    rows: list
    epsilon_used: float


@dataclass
class _Match:
    entry: _Entry


class SMTEquivalenceMatcher:
    """Cache matcher that admits a hit only when Z3 proves the two queries'
    WHERE predicates equivalent (and their non-WHERE structure is identical)."""

    def __init__(self, int_columns: Iterable[str] = ()):
        self.int_columns = frozenset(int_columns)
        # bucket by non-WHERE structure -> list of (parsed, entry)
        self._buckets: dict[tuple, list] = {}

    def add(self, parsed: ParsedQuery, columns, rows, epsilon_used: float) -> None:
        sig = _structural_sig(parsed)
        self._buckets.setdefault(sig, []).append(
            (parsed, _Entry(columns, rows, epsilon_used)))

    def find_match(self, parsed: ParsedQuery) -> Optional[_Match]:
        if z3 is None:
            return None
        bucket = self._buckets.get(_structural_sig(parsed))
        if not bucket:
            return None
        new_where = parsed.ast.find(exp.Where)
        new_where = new_where.this if new_where is not None else None
        for cached, entry in bucket:
            old_where = cached.ast.find(exp.Where)
            old_where = old_where.this if old_where is not None else None
            if predicates_equivalent(new_where, old_where, self.int_columns) is True:
                return _Match(entry)
        return None
