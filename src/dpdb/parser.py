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
    position: int = -1  # index within the SELECT list = result column index


@dataclass
class ParsedQuery:
    raw_sql: str
    ast: exp.Expression
    table: str
    aggregates: list[AggregateInfo] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    where_clause: Optional[str] = None
    where_predicates: list[str] = field(default_factory=list)
    # Top-k support: ORDER BY <result column> [DESC|ASC] LIMIT k. The ordering and
    # truncation are applied as POST-PROCESSING on the noised result, so a top-k
    # query costs the same epsilon as the underlying GROUP BY.
    order_by_position: Optional[int] = None  # result-column index to sort by
    order_desc: bool = True
    limit: Optional[int] = None


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

    # Reject HAVING: it filters groups on a data-dependent aggregate condition,
    # which is an unaccounted private selection (group membership leaks the true
    # aggregate crossing the threshold) and is outside the supported DP subset.
    if ast.args.get("having") is not None:
        raise ParseError("HAVING is not supported in the current version")

    # Extract GROUP BY first so we can validate SELECT against it
    group_by = []
    group_clause = ast.args.get("group")
    if group_clause:
        for g in group_clause.expressions:
            if isinstance(g, exp.Column):
                group_by.append(g.name)
            else:
                group_by.append(g.sql())
    group_by_set = set(group_by)

    # Extract aggregates from SELECT, enforce GROUP BY for non-aggregate columns.
    # Record each aggregate's position in the SELECT list so the noise mechanism
    # can target the correct result column regardless of column ordering.
    aggregates = []
    for pos, select_expr in enumerate(ast.expressions):
        agg = _extract_aggregate(select_expr)
        if agg:
            agg.position = pos
            aggregates.append(agg)
            continue
        # Non-aggregate: must be a column AND must appear in GROUP BY
        col = None
        if isinstance(select_expr, exp.Column):
            col = select_expr
        elif isinstance(select_expr, exp.Alias):
            inner = select_expr.this
            if isinstance(inner, exp.Column):
                col = inner
        if col is None:
            raise ParseError(
                f"Non-aggregate, non-column expression in SELECT: {select_expr.sql()}"
            )
        if col.name not in group_by_set:
            raise ParseError(
                f"Non-aggregate column '{col.name}' in SELECT is not in GROUP BY"
            )

    if not aggregates:
        raise ParseError("At least one aggregate function required (COUNT, SUM, AVG)")

    # Extract WHERE
    where = ast.find(exp.Where)
    where_clause = None
    where_predicates = []
    if where:
        where_clause = where.this.sql()
        where_predicates = _extract_predicates(where.this)

    # Top-k: ORDER BY <result column> [DESC|ASC] LIMIT k. Both are applied as
    # post-processing on the noised result, so no extra privacy budget is charged.
    order_by_position = None
    order_desc = True
    order_clause = ast.args.get("order")
    if order_clause is not None:
        ordered = order_clause.expressions
        if len(ordered) != 1:
            raise ParseError("ORDER BY supports a single key in the DP subset")
        order_desc = bool(ordered[0].args.get("desc"))
        order_by_position = _resolve_select_position(ordered[0].this, ast.expressions)
        if order_by_position is None:
            raise ParseError(
                "ORDER BY must reference a SELECT column/aggregate (top-k)")

    limit = None
    limit_clause = ast.args.get("limit")
    if limit_clause is not None:
        lit = limit_clause.expression
        if not (isinstance(lit, exp.Literal) and lit.is_int):
            raise ParseError("LIMIT must be an integer literal")
        limit = int(lit.this)
        if limit <= 0:
            raise ParseError("LIMIT must be positive")

    return ParsedQuery(
        raw_sql=sql,
        ast=ast,
        table=table_name,
        aggregates=aggregates,
        group_by=group_by,
        where_clause=where_clause,
        where_predicates=sorted(where_predicates),
        order_by_position=order_by_position,
        order_desc=order_desc,
        limit=limit,
    )


def _resolve_select_position(target: exp.Expression,
                             select_exprs: list[exp.Expression]) -> Optional[int]:
    """Map an ORDER BY target to its 0-based result-column index (or None)."""
    if isinstance(target, exp.Literal) and target.is_int:        # ORDER BY 2
        return int(target.this) - 1
    tname = target.name if isinstance(target, exp.Column) else None
    tsql = target.sql()
    for i, se in enumerate(select_exprs):
        if isinstance(se, exp.Alias):
            if tname is not None and se.alias == tname:           # ORDER BY alias
                return i
            inner = se.this
        else:
            inner = se
        if inner.sql() == tsql:                                   # ORDER BY COUNT(*)
            return i
        if tname is not None and isinstance(inner, exp.Column) and inner.name == tname:
            return i                                              # ORDER BY group_key
    return None


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
            return AggregateInfo(
                func=func_name, column=_agg_column(found.this), alias=alias
            )

    if isinstance(node, (exp.Count, exp.Sum, exp.Avg)):
        func_name = type(node).__name__.upper()
        return AggregateInfo(
            func=func_name, column=_agg_column(node.this), alias=alias
        )

    return None


def _agg_column(col_expr: Optional[exp.Expression]) -> Optional[str]:
    """Resolve the column an aggregate is applied to, rejecting out-of-scope args.

    DISTINCT and arbitrary expressions are not part of the supported DP subset
    and must be rejected rather than silently stringified into a bogus column
    name (which previously produced a confidently-labelled but ill-defined
    sensitivity).
    """
    if isinstance(col_expr, exp.Distinct):
        raise ParseError("DISTINCT aggregates are not supported in the current version")
    if col_expr is None or isinstance(col_expr, exp.Star):
        return None
    if isinstance(col_expr, exp.Column):
        return col_expr.name
    raise ParseError(f"Unsupported aggregate argument: {col_expr.sql()}")


def _extract_predicates(node: exp.Expression) -> list[str]:
    """Extract individual predicates from a WHERE clause."""
    if isinstance(node, exp.And):
        left = _extract_predicates(node.left)
        right = _extract_predicates(node.right)
        return left + right
    return [node.sql()]
