"""Main DP middleware orchestrator."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from sqlglot import exp

from dpdb.analyzer import SensitivityResult, analyze_sensitivity
from dpdb.budget import AllocationStrategy, BudgetExhausted, BudgetLedger
from dpdb.config import Config
from dpdb.db import Database, create_database
from dpdb.mechanisms import laplace_mechanism
from dpdb.parser import ParsedQuery, parse_query


class ExecutionMode(str, Enum):
    EXACT = "exact"
    NAIVE_DP = "naive_dp"
    WORKLOAD_DP = "workload_dp"
    SEMANTIC_DP = "semantic_dp"
    TEMPORAL_DP = "temporal_dp"     # workload-aware + staleness/update model
    PREDICTIVE_DP = "predictive_dp" # workload-aware + model-driven adaptive epsilon


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    epsilon_used: float
    cache_hit: bool
    latency_ms: float
    sensitivities: list[SensitivityResult] = field(default_factory=list)
    error: Optional[str] = None


class DPMiddleware:
    """Differentially private SQL middleware for repeated aggregate queries."""

    def __init__(
        self,
        config: Config,
        mode: ExecutionMode = ExecutionMode.EXACT,
        db: Optional[Database] = None,
        staleness_tolerance: float = float("inf"),
        update_rate: float = 0.0,
        update_invalidation_prob: float = 0.0,
        update_seed: int = 0,
        predictive_k_total: int = 100,
        predictive_warmup_fraction: float = 0.1,
    ):
        self.config = config
        self.mode = mode
        self.db = db or create_database(config)
        self.predictor = None

        if mode == ExecutionMode.NAIVE_DP:
            self.budget = BudgetLedger(
                config.privacy.total_epsilon, AllocationStrategy.NAIVE
            )
        elif mode == ExecutionMode.WORKLOAD_DP:
            self.budget = BudgetLedger(
                config.privacy.total_epsilon, AllocationStrategy.WORKLOAD_AWARE
            )
        elif mode == ExecutionMode.TEMPORAL_DP:
            self.budget = BudgetLedger(
                config.privacy.total_epsilon,
                AllocationStrategy.WORKLOAD_AWARE,
                staleness_tolerance=staleness_tolerance,
                update_rate=update_rate,
                update_invalidation_prob=update_invalidation_prob,
                update_seed=update_seed,
            )
        elif mode == ExecutionMode.PREDICTIVE_DP:
            from dpdb.predictive import PredictiveAllocator, PredictiveConfig
            self.predictor = PredictiveAllocator(PredictiveConfig(
                total_budget=config.privacy.total_epsilon,
                k_total=predictive_k_total,
                warmup_fraction=predictive_warmup_fraction,
            ))
            self.budget = BudgetLedger(
                config.privacy.total_epsilon,
                AllocationStrategy.WORKLOAD_AWARE,
            )
        elif mode == ExecutionMode.SEMANTIC_DP:
            from dpdb.semantic import SemanticMatcher
            matcher = SemanticMatcher(
                tk_threshold=0.95,
                emb_threshold=0.98,
                use_embedding=True,
            )
            self.budget = BudgetLedger(
                config.privacy.total_epsilon,
                AllocationStrategy.SEMANTIC_AWARE,
                semantic_matcher=matcher,
            )
        else:
            self.budget = None

    def execute(self, sql: str, epsilon: Optional[float] = None) -> QueryResult:
        """Execute a SQL query, optionally with differential privacy."""
        start = time.perf_counter()

        if self.mode == ExecutionMode.EXACT:
            return self._execute_exact(sql, start)

        try:
            parsed = parse_query(sql)
        except Exception as e:
            return QueryResult(
                columns=[], rows=[], epsilon_used=0.0,
                cache_hit=False, latency_ms=_elapsed(start),
                error=str(e),
            )

        # Predictive mode overrides the per-query epsilon
        if self.mode == ExecutionMode.PREDICTIVE_DP and self.predictor is not None:
            eps = self.predictor.next_epsilon(parsed)
        else:
            # Fall back to the default ONLY when epsilon is omitted (None). A
            # supplied epsilon must be a finite positive number: epsilon=0 must
            # NOT silently become the default, and NaN/inf/negative must not be
            # allowed to corrupt the budget ledger.
            eps = self.config.privacy.default_query_epsilon if epsilon is None else epsilon
            if not (isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0):
                return QueryResult(
                    columns=[], rows=[], epsilon_used=0.0,
                    cache_hit=False, latency_ms=_elapsed(start),
                    error=f"Invalid epsilon: must be finite and positive (got {epsilon!r})",
                )

        # Cache hits short-circuit before budget allocation: they cost ε=0
        # and must succeed even when the predictor says the budget is dry.
        if self.budget:
            cached = self.budget.try_cache(parsed)
            if cached is not None:
                return QueryResult(
                    columns=cached.columns,
                    rows=cached.rows,
                    epsilon_used=0.0,
                    cache_hit=True,
                    latency_ms=_elapsed(start),
                )

        # Cache miss — now we need a positive budget to release a fresh noisy answer.
        if self.mode == ExecutionMode.PREDICTIVE_DP and eps <= 0:
            return QueryResult(
                columns=[], rows=[], epsilon_used=0.0,
                cache_hit=False, latency_ms=_elapsed(start),
                error="Predictive allocator returned zero budget",
            )

        # Analyze sensitivity
        try:
            sensitivities = analyze_sensitivity(parsed, self.config)
        except Exception as e:
            return QueryResult(
                columns=[], rows=[], epsilon_used=0.0,
                cache_hit=False, latency_ms=_elapsed(start),
                error=str(e),
            )

        # Allocate budget
        try:
            allocated_eps = self.budget.allocate(parsed, eps)
        except BudgetExhausted as e:
            return QueryResult(
                columns=[], rows=[], epsilon_used=0.0,
                cache_hit=False, latency_ms=_elapsed(start),
                error=str(e),
            )

        # Execute true query. For ANY ORDER BY (with or without LIMIT) we must
        # fetch the FULL histogram and order/select on the NOISED values: letting
        # the DB apply ORDER BY on the TRUE counts would leak the true ranking
        # through the row order even when the values are noised. We therefore
        # strip ORDER BY and LIMIT whenever either is present, then re-apply both
        # as post-processing on the noised result (no extra budget). LIMIT without
        # ORDER BY is rejected at parse time. We also clamp SUM/AVG arguments to
        # their public domain bound B_c IN the executed SQL, so the realized
        # sensitivity can never exceed the Delta f the noise is calibrated to
        # (otherwise out-of-bound data would silently break the DP guarantee).
        base = parsed.ast.copy()
        stripped = parsed.order_by_position is not None or parsed.limit is not None
        if stripped:
            base.set("order", None)
            base.set("limit", None)
        clamped = self._clamp_aggregate_columns(base, parsed.table)
        exec_sql = base.sql(dialect="postgres") if (stripped or clamped) else sql
        columns, true_rows = self.db.execute_with_columns(exec_sql)

        # Add noise to each row's aggregate columns
        noisy_rows = self._add_noise(parsed, true_rows, sensitivities, allocated_eps)

        # Post-processing on the noised result: ORDER BY the noisy value, then
        # (if present) LIMIT. Ordering is applied for ANY ORDER BY, not only the
        # top-k case, so the released row order never reflects the true ranking.
        if parsed.order_by_position is not None:
            noisy_rows = sorted(
                noisy_rows, key=lambda r: r[parsed.order_by_position],
                reverse=parsed.order_desc)
        if parsed.limit is not None:
            noisy_rows = noisy_rows[:parsed.limit]

        # Cache result
        if self.budget:
            self.budget.store_result(parsed, columns, noisy_rows, allocated_eps)
        # Predictive bookkeeping
        if self.predictor is not None:
            self.predictor.note_release(parsed, allocated_eps)

        return QueryResult(
            columns=columns,
            rows=noisy_rows,
            epsilon_used=allocated_eps,
            cache_hit=False,
            latency_ms=_elapsed(start),
            sensitivities=sensitivities,
        )

    def _clamp_aggregate_columns(self, ast, table: str) -> bool:
        """Rewrite SUM(c)/AVG(c) -> SUM(GREATEST(LEAST(c, B_c), 0)) for every
        column c with a configured public bound B_c, so the executed aggregate
        provably respects |c| <= B_c. Without this clamp, a value exceeding the
        configured bound would make the true sensitivity larger than the
        Delta f = B_c the Laplace noise is calibrated to, breaking the epsilon-DP
        guarantee. All configured columns are non-negative quantities, so the
        clamp range is [0, B_c]. The original output column name is preserved.
        Returns True if any aggregate was rewritten.
        """
        changed = False
        for agg in list(ast.find_all(exp.Sum, exp.Avg)):
            col = agg.this
            if not isinstance(col, exp.Column):
                continue
            bound = self.config.get_bound(table, col.name)
            if bound is None:
                continue
            orig_name = agg.sql(dialect="postgres")
            clamp = exp.Greatest(
                this=exp.Least(
                    this=col.copy(),
                    expressions=[exp.Literal.number(bound)],
                ),
                expressions=[exp.Literal.number(0)],
            )
            # Preserve SQL NULL semantics: GREATEST/LEAST treat NULL as a missing
            # value (LEAST(NULL,B)=B), which would turn a NULL into B and corrupt
            # SUM/AVG. Guard with CASE so NULL passes through untouched and is
            # excluded from the aggregate exactly as in the unclamped query.
            agg.set("this", exp.Case(
                ifs=[exp.If(
                    this=exp.Is(this=col.copy(), expression=exp.Null()),
                    true=exp.Null(),
                )],
                default=clamp,
            ))
            # Keep the released column's name stable for bare (un-aliased) aggregates.
            if not isinstance(agg.parent, exp.Alias):
                agg.replace(exp.alias_(agg.copy(), orig_name, quoted=True))
            changed = True
        return changed

    def _execute_exact(self, sql: str, start: float) -> QueryResult:
        columns, rows = self.db.execute_with_columns(sql)
        return QueryResult(
            columns=columns,
            rows=rows,
            epsilon_used=0.0,
            cache_hit=False,
            latency_ms=_elapsed(start),
        )

    def _add_noise(
        self,
        parsed: ParsedQuery,
        true_rows: list[tuple],
        sensitivities: list[SensitivityResult],
        epsilon: float,
    ) -> list[tuple]:
        """Add calibrated Laplace noise to aggregate columns in each row.

        For queries with GROUP BY, each group's aggregates get independent noise.
        Budget epsilon is split equally among aggregates in a single query
        (parallel composition when aggregates are over different columns,
         sequential composition as a conservative default).
        """
        n_aggs = len(sensitivities)
        if n_aggs == 0:
            return true_rows

        # Split epsilon among aggregates (sequential composition, conservative)
        eps_per_agg = epsilon / n_aggs

        # Map each aggregate to its actual result-column index using the position
        # recorded by the parser. Do NOT assume group-by columns come first: with
        # an aggregate-first SELECT (e.g. `SELECT COUNT(*), grp ... GROUP BY grp`)
        # the positional assumption would noise the group key and release the true
        # aggregate in the clear — a differential-privacy violation.
        noisy_rows = []
        for row in true_rows:
            row_list = list(row)
            for i, sens in enumerate(sensitivities):
                col_idx = parsed.aggregates[i].position
                true_val = float(row_list[col_idx]) if row_list[col_idx] is not None else 0.0

                # AVG is released as a single conservative Laplace mechanism with
                # sensitivity = column bound B_c (worst case: group of size one).
                # A SUM+COUNT decomposition that amortizes noise by group size is
                # future work (see report Future Work); it is NOT done here.
                noisy_val = laplace_mechanism(true_val, sens.sensitivity, eps_per_agg)

                # Post-processing: COUNT should be non-negative integer
                if sens.func == "COUNT":
                    noisy_val = max(0, round(noisy_val))

                row_list[col_idx] = noisy_val
            noisy_rows.append(tuple(row_list))

        return noisy_rows

    def budget_summary(self) -> Optional[dict]:
        if self.budget:
            return self.budget.summary()
        return None

    def remaining_budget(self) -> float:
        if self.budget:
            return self.budget.remaining
        return float("inf")


def _elapsed(start: float) -> float:
    return (time.perf_counter() - start) * 1000
