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
    # FK-join support (single 2-table inner equi-join). When is_join is True the
    # privacy unit is a parent ENTITY and the COUNT sensitivity is the public FK
    # multiplicity d_max, not 1 (resolved against config in the analyzer).
    is_join: bool = False
    join_table: Optional[str] = None
    join_on: Optional[str] = None
    tables: list[str] = field(default_factory=list)
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

    # Positive whitelist for the structure: reject anything that can amplify or
    # hide a row's contribution and thus invalidate the sensitivity bound.
    if ast.args.get("with") is not None or ast.find(exp.With, exp.CTE) is not None:
        raise ParseError(
            "WITH/CTE is not supported (a self-union CTE can amplify each row's "
            "contribution while the sensitivity is still computed as 1)")
    if ast.find(exp.Union, exp.Intersect, exp.Except) is not None:
        raise ParseError("Set operations (UNION/INTERSECT/EXCEPT) are not supported")
    if ast.args.get("distinct") is not None:
        raise ParseError("SELECT DISTINCT is not supported")
    if ast.args.get("offset") is not None or ast.find(exp.Offset) is not None:
        raise ParseError(
            "OFFSET is not supported (it would be applied on the true histogram "
            "before noising)")

    # Reject subqueries / derived tables
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

    # Single FK join support: exactly one INNER equi-join of two tables. The
    # privacy unit is the parent entity and the COUNT sensitivity is the public
    # FK multiplicity d_max (resolved in the analyzer). OUTER/CROSS joins, more
    # than one join, and SUM/AVG over a join are rejected as out of the safe
    # envelope (a row's contribution there can exceed d_max or d_max*B_c in ways
    # this conservative bound does not yet cover -- future work).
    joins = list(ast.find_all(exp.Join))
    is_join = False
    join_table = None
    join_on = None
    if joins:
        if len(joins) != 1:
            raise ParseError("At most one JOIN is supported (single FK join only)")
        j = joins[0]
        side = (j.args.get("side") or "").upper()
        kind = (j.args.get("kind") or "").upper()
        if side in ("LEFT", "RIGHT", "FULL") or kind == "CROSS":
            raise ParseError("Only INNER FK joins are supported (no OUTER/CROSS join)")
        jt = j.find(exp.Table)
        if jt is None:
            raise ParseError("Could not identify the joined table")
        join_table = jt.name
        on = j.args.get("on")
        if on is None:
            raise ParseError("JOIN requires an ON equality condition on the foreign key")
        join_on = on.sql()
        is_join = True

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

    # Over a join, only COUNT is supported: its sensitivity is exactly the FK
    # multiplicity d_max. SUM/AVG over a join would need a per-table column-bound
    # clamp times d_max, which is not yet wired through the executor (future work).
    if is_join and any(a.func != "COUNT" for a in aggregates):
        raise ParseError(
            "Only COUNT is supported over a JOIN in the current version "
            "(SUM/AVG over joins are future work)")

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

    # A LIMIT without an explicit ORDER BY is an ill-defined top-k: the database
    # would return an arbitrary subset, and the DP guarantee for top-k relies on
    # ranking the NOISED aggregate. Require the analyst to state the order, and
    # require that order to target an AGGREGATE column (ranking/truncating on a
    # public group key is not a report-noisy-max selection).
    if limit is not None:
        if order_by_position is None:
            raise ParseError(
                "LIMIT requires an explicit ORDER BY on a SELECT aggregate "
                "(top-k must rank on the noised value)")
        agg_positions = {a.position for a in aggregates}
        if order_by_position not in agg_positions:
            raise ParseError(
                "top-k ORDER BY must target an aggregate column, not a group key")

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
        is_join=is_join,
        join_table=join_table,
        join_on=join_on,
        tables=[table_name] + ([join_table] if join_table else []),
    )


def _resolve_select_position(target: exp.Expression,
                             select_exprs: list[exp.Expression]) -> Optional[int]:
    """Map an ORDER BY target to its 0-based result-column index (or None)."""
    if isinstance(target, exp.Literal) and target.is_int:        # ORDER BY 2
        n = int(target.this)
        if 1 <= n <= len(select_exprs):                          # reject 0 / out-of-range
            return n - 1
        return None
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
    """Extract aggregate info from a SELECT expression.

    The projection must be the aggregate ITSELF (optionally aliased), never an
    expression that merely *contains* one. Earlier this used ``node.find(agg)``,
    which matched the inner SUM of ``1000*SUM(age)`` or ``COUNT(*) OVER ()`` and
    then calibrated noise to the bare aggregate's sensitivity (Delta f=B_c) while
    the released value actually had sensitivity ``1000*B_c`` (or leaked the whole
    table via the window). We now require a direct, un-wrapped aggregate and
    reject any nested/over-windowed/filtered aggregate explicitly.
    """
    alias = None
    if isinstance(node, exp.Alias):
        alias = node.alias
        node = node.this

    # A bare, directly-applied aggregate is the ONLY accepted aggregate form.
    if isinstance(node, (exp.Count, exp.Sum, exp.Avg)) and not isinstance(node, exp.Window):
        # Reject a windowed aggregate (e.g. SUM(x) OVER (...)) that sqlglot may
        # attach as a 'over' arg on the function node itself.
        if node.args.get("over") is not None:
            raise ParseError(
                f"Window/OVER aggregates are not supported: {node.sql()}")
        func_name = type(node).__name__.upper()
        if func_name not in SUPPORTED_AGGREGATES:
            return None
        return AggregateInfo(
            func=func_name, column=_agg_column(node.this), alias=alias
        )

    # An aggregate nested inside another expression (arithmetic, window, FILTER,
    # COALESCE, CASE, ...) has a DIFFERENT sensitivity than the bare aggregate and
    # must be rejected, not silently mis-calibrated or treated as a group column.
    if isinstance(node, exp.Window) or node.find(exp.AggFunc) is not None:
        raise ParseError(
            f"Aggregate must be a direct SELECT item (no arithmetic, window, "
            f"FILTER, COALESCE, or CASE wrapping): {node.sql()}")

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
