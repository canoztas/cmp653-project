"""SQL parsing and validation for the supported DP subset."""

from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import exp


SUPPORTED_AGGREGATES = {"COUNT", "SUM", "AVG"}


@dataclass
class AggregateInfo:
    func: str          # COUNT, SUM, AVG
    column: Optional[str]  # None for COUNT(*)
    alias: Optional[str]


@dataclass
class ParsedQuery:
    raw_sql: str
    ast: exp.Expression
    table: str
    aggregates: list[AggregateInfo] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    where_clause: Optional[str] = None
    where_predicates: list[str] = field(default_factory=list)


class ParseError(Exception):
    pass


def parse_query(sql: str) -> ParsedQuery:
    """Parse and validate a SQL query against the supported DP subset."""
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception as e:
        raise ParseError(f"SQL parse error: {e}")

    if len(statements) != 1:
        raise ParseError("Exactly one SQL statement required")

    ast = statements[0]
    if not isinstance(ast, exp.Select):
        raise ParseError("Only SELECT statements are supported")

    # Reject subqueries
    subqueries = list(ast.find_all(exp.Subquery))
    if subqueries:
        raise ParseError("Subqueries are not supported")

    # Extract table name
    from_clause = ast.find(exp.From)
    if from_clause is None:
        raise ParseError("FROM clause required")

    table_expr = from_clause.find(exp.Table)
    if table_expr is None:
        raise ParseError("Could not identify table")
    table_name = table_expr.name

    # Reject JOINs for now (single-table only)
    joins = list(ast.find_all(exp.Join))
    if joins:
        raise ParseError("JOINs are not supported in the current version")

    # Extract aggregates from SELECT
    aggregates = []
    for select_expr in ast.expressions:
        agg = _extract_aggregate(select_expr)
        if agg:
            aggregates.append(agg)
        else:
            # Allow GROUP BY columns in SELECT
            if isinstance(select_expr, (exp.Column, exp.Alias)):
                col = select_expr.find(exp.Column)
                if col is not None:
                    continue
            raise ParseError(
                f"Non-aggregate, non-group-by expression in SELECT: {select_expr.sql()}"
            )

    if not aggregates:
        raise ParseError("At least one aggregate function required (COUNT, SUM, AVG)")

    # Extract GROUP BY
    group_by = []
    group_clause = ast.args.get("group")
    if group_clause:
        for g in group_clause.expressions:
            if isinstance(g, exp.Column):
                group_by.append(g.name)
            else:
                group_by.append(g.sql())

    # Extract WHERE
    where = ast.find(exp.Where)
    where_clause = None
    where_predicates = []
    if where:
        where_clause = where.this.sql()
        where_predicates = _extract_predicates(where.this)

    return ParsedQuery(
        raw_sql=sql,
        ast=ast,
        table=table_name,
        aggregates=aggregates,
        group_by=group_by,
        where_clause=where_clause,
        where_predicates=sorted(where_predicates),
    )


def _extract_aggregate(node: exp.Expression) -> Optional[AggregateInfo]:
    """Extract aggregate info from a SELECT expression."""
    alias = None
    if isinstance(node, exp.Alias):
        alias = node.alias
        node = node.this

    for agg_type in (exp.Count, exp.Sum, exp.Avg):
        found = node.find(agg_type)
        if found is not None:
            func_name = type(found).__name__.upper()
            if func_name not in SUPPORTED_AGGREGATES:
                continue
            col_expr = found.this
            if isinstance(col_expr, exp.Star):
                column = None
            elif isinstance(col_expr, exp.Column):
                column = col_expr.name
            else:
                column = col_expr.sql() if col_expr else None
            return AggregateInfo(func=func_name, column=column, alias=alias)

    if isinstance(node, (exp.Count, exp.Sum, exp.Avg)):
        func_name = type(node).__name__.upper()
        col_expr = node.this
        if isinstance(col_expr, exp.Star):
            column = None
        elif isinstance(col_expr, exp.Column):
            column = col_expr.name
        else:
            column = col_expr.sql() if col_expr else None
        return AggregateInfo(func=func_name, column=column, alias=alias)

    return None


def _extract_predicates(node: exp.Expression) -> list[str]:
    """Extract individual predicates from a WHERE clause."""
    if isinstance(node, exp.And):
        left = _extract_predicates(node.left)
        right = _extract_predicates(node.right)
        return left + right
    return [node.sql()]
