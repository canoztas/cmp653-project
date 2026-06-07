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
                 semantic_matcher=None,
                 staleness_tolerance: float = float("inf"),
                 update_rate: float = 0.0,
                 update_invalidation_prob: float = 0.0,
                 update_seed: int = 0):
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
        # Temporal regime (R3 extension)
        self.staleness_tolerance = staleness_tolerance  # tau (in logical query units)
        self.update_rate = update_rate                  # lambda (per logical step)
        self.update_invalidation_prob = update_invalidation_prob
        self.update_seed = update_seed                  # per-trial seed for update RNG
        self._logical_time = 0                          # incremented per query
        self.expired_evictions = 0
        self.update_evictions = 0
        self._rng = None  # initialized lazily for update simulation

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    def _tick_clock(self):
        """Advance logical time; simulate update arrivals if update_rate > 0."""
        self._logical_time += 1
        if self.update_rate > 0 and self.update_invalidation_prob > 0:
            import random
            if self._rng is None:
                # Seed from the per-trial update_seed so independent trials draw
                # independent update streams (a fixed Random(42) made every trial
                # share one update realization, collapsing the reported variance).
                self._rng = random.Random(self.update_seed)
            # Per-step Bernoulli: probability of an update event in this step
            if self._rng.random() < self.update_rate:
                # An update occurred; invalidate a random fraction of cache
                for t_hash, params in self._cache.items():
                    for p_hash, entry in list(params.items()):
                        if not entry.invalidated and self._rng.random() < self.update_invalidation_prob:
                            entry.invalidated = True
                            self.update_evictions += 1

    def _is_stale(self, entry: CachedResult) -> bool:
        """Check if a cached entry has exceeded staleness tolerance or been invalidated."""
        if entry.invalidated:
            return True
        age = self._logical_time - entry.issued_at
        return age > self.staleness_tolerance

    def try_cache(self, parsed: ParsedQuery) -> Optional[CachedResult]:
        """Check cache for this query.

        - NAIVE: never caches.
        - WORKLOAD_AWARE: exact template + parameter match (with optional temporal check).
        - SEMANTIC_AWARE: exact match first, then semantic similarity match.
        """
        self._tick_clock()
        if self.strategy == AllocationStrategy.NAIVE:
            return None

        t_hash = template_hash(extract_template(parsed))
        p_hash = param_hash(parsed)

        # L1: Exact match (template + parameters)
        template_cache = self._cache.get(t_hash)
        if template_cache is not None:
            cached = template_cache.get(p_hash)
            if cached is not None:
                if self._is_stale(cached):
                    # Entry expired or invalidated - evict and treat as miss
                    del template_cache[p_hash]
                    self.expired_evictions += 1
                else:
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

        # Track distinct EXACT species (template + WHERE literals), the unit the
        # budget is actually charged per, so summary()['unique_templates'] equals
        # u_k. Keying by the structural template alone under-counts: three queries
        # differing only in a WHERE literal each cost epsilon but would collapse
        # to a single structural template.
        species = full_query_hash(parsed)
        self._template_counts[species] = self._template_counts.get(species, 0) + 1

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
            issued_at=self._logical_time,
            invalidated=False,
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
            "expired_evictions": self.expired_evictions,
            "update_evictions": self.update_evictions,
            "logical_time": self._logical_time,
        }
