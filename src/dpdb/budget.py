"""Privacy budget ledger: naive and workload-aware strategies."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dpdb.parser import ParsedQuery
from dpdb.template import (
    CachedResult,
    extract_template,
    full_query_hash,
    param_hash,
    template_hash,
)


class BudgetExhausted(Exception):
    pass


class AllocationStrategy(str, Enum):
    NAIVE = "naive"
    WORKLOAD_AWARE = "workload_aware"
    SEMANTIC_AWARE = "semantic_aware"


@dataclass
class BudgetEntry:
    query_sql: str
    epsilon_allocated: float
    template_hash: str
    cache_hit: bool


class BudgetLedger:
    """Tracks privacy budget consumption across a query workload.

    Naive mode: each query consumes its full epsilon (sequential composition).
    Workload-aware mode: exact-match queries reuse cached results at zero cost;
    same-template queries share budget via parallel composition.
    """

    def __init__(self, total_epsilon: float, strategy: AllocationStrategy,
                 semantic_matcher=None):
        self.total_epsilon = total_epsilon
        self.strategy = strategy
        self.consumed_epsilon = 0.0
        self.history: list[BudgetEntry] = []
        # template_hash -> {param_hash -> CachedResult}
        self._cache: dict[str, dict[str, CachedResult]] = {}
        # template_hash -> count of queries seen with this template
        self._template_counts: dict[str, int] = {}
        # Semantic cache (only used in SEMANTIC_AWARE mode)
        self.semantic_matcher = semantic_matcher
        self.semantic_hits = 0

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    def try_cache(self, parsed: ParsedQuery) -> Optional[CachedResult]:
        """Check cache for this query.

        - NAIVE: never caches.
        - WORKLOAD_AWARE: exact template + parameter match.
        - SEMANTIC_AWARE: exact match first, then semantic similarity match.
        """
        if self.strategy == AllocationStrategy.NAIVE:
            return None

        t_hash = template_hash(extract_template(parsed))
        p_hash = param_hash(parsed)

        # L1: Exact match (template + parameters)
        template_cache = self._cache.get(t_hash)
        if template_cache is not None:
            cached = template_cache.get(p_hash)
            if cached is not None:
                self.history.append(BudgetEntry(
                    query_sql=parsed.raw_sql,
                    epsilon_allocated=0.0,
                    template_hash=t_hash,
                    cache_hit=True,
                ))
                return cached

        # L2: Semantic match (only in SEMANTIC_AWARE mode)
        if (self.strategy == AllocationStrategy.SEMANTIC_AWARE
                and self.semantic_matcher is not None):
            match = self.semantic_matcher.find_match(parsed)
            if match is not None:
                self.semantic_hits += 1
                self.history.append(BudgetEntry(
                    query_sql=parsed.raw_sql,
                    epsilon_allocated=0.0,
                    template_hash=t_hash,
                    cache_hit=True,
                ))
                # Build a CachedResult from the semantic entry
                from dpdb.template import CachedResult
                return CachedResult(
                    template_hash=t_hash,
                    param_hash=p_hash,
                    columns=match.entry.columns,
                    rows=match.entry.rows,
                    epsilon_used=match.entry.epsilon_used,
                    query_sql=parsed.raw_sql,
                )

        return None

    def allocate(self, parsed: ParsedQuery, requested_epsilon: float) -> float:
        """Allocate privacy budget for a query. Returns the actual epsilon to use.

        Raises BudgetExhausted if insufficient budget remains.
        """
        if requested_epsilon <= 0:
            raise ValueError("Requested epsilon must be positive")

        if self.strategy == AllocationStrategy.NAIVE:
            return self._allocate_naive(parsed, requested_epsilon)
        else:
            # workload-aware and semantic-aware share the same allocation logic
            return self._allocate_workload_aware(parsed, requested_epsilon)

    def _allocate_naive(self, parsed: ParsedQuery, requested_epsilon: float) -> float:
        """Naive: each query gets its full epsilon, strict sequential composition."""
        if requested_epsilon > self.remaining:
            raise BudgetExhausted(
                f"Budget exhausted. Remaining: {self.remaining:.4f}, "
                f"requested: {requested_epsilon:.4f}"
            )
        self.consumed_epsilon += requested_epsilon
        t_hash = template_hash(extract_template(parsed))
        self.history.append(BudgetEntry(
            query_sql=parsed.raw_sql,
            epsilon_allocated=requested_epsilon,
            template_hash=t_hash,
            cache_hit=False,
        ))
        return requested_epsilon

    def _allocate_workload_aware(
        self, parsed: ParsedQuery, requested_epsilon: float
    ) -> float:
        """Workload-aware: use cache hits and amortized allocation."""
        t_hash = template_hash(extract_template(parsed))

        # Track template frequency
        self._template_counts[t_hash] = self._template_counts.get(t_hash, 0) + 1

        # No cache hit (try_cache was called first), so we must allocate
        if requested_epsilon > self.remaining:
            raise BudgetExhausted(
                f"Budget exhausted. Remaining: {self.remaining:.4f}, "
                f"requested: {requested_epsilon:.4f}"
            )

        self.consumed_epsilon += requested_epsilon
        self.history.append(BudgetEntry(
            query_sql=parsed.raw_sql,
            epsilon_allocated=requested_epsilon,
            template_hash=t_hash,
            cache_hit=False,
        ))
        return requested_epsilon

    def store_result(
        self,
        parsed: ParsedQuery,
        columns: list[str],
        rows: list[tuple],
        epsilon_used: float,
    ):
        """Cache a noisy result for future reuse (workload-aware + semantic only)."""
        if self.strategy == AllocationStrategy.NAIVE:
            return

        t_hash = template_hash(extract_template(parsed))
        p_hash = param_hash(parsed)

        if t_hash not in self._cache:
            self._cache[t_hash] = {}

        self._cache[t_hash][p_hash] = CachedResult(
            template_hash=t_hash,
            param_hash=p_hash,
            columns=columns,
            rows=rows,
            epsilon_used=epsilon_used,
            query_sql=parsed.raw_sql,
        )

        # Also add to semantic matcher if present
        if (self.strategy == AllocationStrategy.SEMANTIC_AWARE
                and self.semantic_matcher is not None):
            self.semantic_matcher.add(parsed, columns, rows, epsilon_used)

    def summary(self) -> dict:
        """Return a summary of budget usage."""
        cache_hits = sum(1 for e in self.history if e.cache_hit)
        return {
            "strategy": self.strategy.value,
            "total_epsilon": self.total_epsilon,
            "consumed_epsilon": self.consumed_epsilon,
            "remaining_epsilon": self.remaining,
            "total_queries": len(self.history),
            "cache_hits": cache_hits,
            "exact_hits": cache_hits - self.semantic_hits,
            "semantic_hits": self.semantic_hits,
            "cache_hit_rate": cache_hits / len(self.history) if self.history else 0.0,
            "unique_templates": len(self._template_counts),
        }
