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
    """Deterministic hash of a template string."""
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def param_hash(parsed: ParsedQuery) -> str:
    """Hash of the literal values in WHERE (captures the specific parameters)."""
    literals = []
    where = parsed.ast.find(exp.Where)
    if where:
        for lit in where.find_all(exp.Literal):
            literals.append(lit.this)
    key = "|".join(str(l) for l in literals)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


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
