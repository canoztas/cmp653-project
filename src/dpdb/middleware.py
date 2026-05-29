"""Main DP middleware orchestrator."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

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
            eps = epsilon or self.config.privacy.default_query_epsilon

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

        # Execute true query
        columns, true_rows = self.db.execute_with_columns(sql)

        # Add noise to each row's aggregate columns
        noisy_rows = self._add_noise(parsed, true_rows, sensitivities, allocated_eps)

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
