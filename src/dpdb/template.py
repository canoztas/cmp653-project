"""Query template extraction and matching for workload-aware caching."""

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from dpdb.parser import ParsedQuery


@dataclass
class CachedResult:
    template_hash: str
    param_hash: str
    columns: list[str]
    rows: list[tuple]
    epsilon_used: float
    query_sql: str
    # Temporal metadata (R3 extension)
    issued_at: float = 0.0    # logical time when this noisy result was released
    invalidated: bool = False  # set True by an update event affecting this entry


def extract_template(parsed: ParsedQuery) -> str:
    """Extract a canonical template by replacing WHERE literals with placeholders.

    Two queries with the same structure but different literal values in WHERE
    produce the same template.
    """
    ast_copy = parsed.ast.copy()
    # Replace all literal values in WHERE with '?'
    for literal in ast_copy.find_all(exp.Literal):
        # Only replace literals inside WHERE predicates
        parent = literal.parent
        if _is_in_where(literal, ast_copy):
            literal.replace(exp.Placeholder())

    normalized = ast_copy.sql(dialect="postgres", pretty=False)
    return normalized


def template_hash(template: str) -> str:
    """Deterministic, collision-resistant hash of a template string (full SHA-256)."""
    return hashlib.sha256(template.encode()).hexdigest()


def param_hash(parsed: ParsedQuery) -> str:
    """Collision-resistant hash of the WHERE literal values (type + value).

    Each literal is serialized as ``<type>:<len>:<value>`` and the parts joined
    with a control-character separator, so neither the string/number type nor a
    value that happens to contain the separator can produce a colliding key for
    the *serialization*. A naive ``"|".join(values)`` collides: ``education='a|b'
    AND sex='c'`` and ``education='a' AND sex='b|c'`` both flatten to ``a|b|c``;
    likewise ``age=1`` (number) and ``age='1'`` (string). The type tag and length
    prefix make the encoding injective. We then take the full SHA-256 digest
    (not a 64-bit truncation), so it is collision-*resistant* rather than truly
    collision-free; callers that need an exact decision can compare the canonical
    serialized keys directly.
    """
    parts = []
    where = parsed.ast.find(exp.Where)
    if where:
        for lit in where.find_all(exp.Literal):
            val = str(lit.this)
            typ = "s" if lit.is_string else "n"
            parts.append(f"{typ}:{len(val)}:{val}")
    key = "\x1f".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()


def full_query_hash(parsed: ParsedQuery) -> str:
    """Hash combining template + parameters. Exact match = identical query."""
    t = extract_template(parsed)
    p = param_hash(parsed)
    return f"{template_hash(t)}:{p}"


def is_exact_match(q1: ParsedQuery, q2: ParsedQuery) -> bool:
    """Check if two queries are identical (same template + same params)."""
    return full_query_hash(q1) == full_query_hash(q2)


def is_same_template(q1: ParsedQuery, q2: ParsedQuery) -> bool:
    """Check if two queries share the same template (structure)."""
    t1 = extract_template(q1)
    t2 = extract_template(q2)
    return template_hash(t1) == template_hash(t2)


def _is_in_where(node: exp.Expression, root: exp.Expression) -> bool:
    """Check if a node is inside the WHERE clause."""
    current = node
    while current is not None:
        if isinstance(current, exp.Where):
            return True
        current = current.parent
    return False
