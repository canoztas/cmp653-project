"""Predictive budget allocator built on the R2 analytical model.

Mechanism:
  1. Maintain an online frequency counter over observed template hashes.
  2. After k_warmup queries, treat the empirical frequency vector as an
     estimate of the workload's template distribution {p_i}.
  3. Use the R2 model (Proposition 1) to predict E[u_total] under that
     distribution for the full workload of length k_total.
  4. Subtract the unique templates seen so far to get u_remaining.
  5. Allocate remaining budget evenly across the predicted u_remaining
     cache misses, so analyst budget is spread over PREDICTED unique
     queries instead of literal query slots.

This is novel relative to existing DP-SQL systems (PINQ, PrivateSQL,
Chorus, DOP-SQL): they all use a fixed per-query eps_q. Our allocator
adapts eps_q in flight, using the R2 model as the predictive backbone.

Privacy correctness: the allocator only adapts the *budget split*, not
the noise calibration. Each released noisy answer still satisfies its
own eps_i-DP guarantee. Total privacy loss = sum of eps_i across cache
misses, which we cap at total_epsilon by construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dpdb.model import expected_unique_queries
from dpdb.parser import ParsedQuery
from dpdb.predictors import predict_smoothed_gt
from dpdb.template import extract_template, template_hash


@dataclass
class PredictiveConfig:
    total_budget: float        # eps_total for the workload
    k_total: int               # expected length of the workload (analyst declares)
    warmup_fraction: float = 0.1  # warmup phase uses fallback rule
    min_warmup: int = 5
    floor_eps: float = 0.01    # never go below this per-query eps
    pseudocount: float = 0.5   # Laplace smoothing on empirical frequencies
    estimator: str = "plugin"  # "plugin" (occupancy) or "smoothed_gt" (unseen-species)


class PredictiveAllocator:
    """Online predictive budget allocator.

    Usage:
        alloc = PredictiveAllocator(PredictiveConfig(...))
        for query in workload:
            eps_q = alloc.next_epsilon(parsed_query)
            if eps_q > 0:
                middleware.execute(query, epsilon=eps_q)
                alloc.note_release(parsed_query, eps_q)
            else:
                # cache hit or budget exhausted
                ...
    """

    def __init__(self, config: PredictiveConfig):
        self.cfg = config
        self.queries_seen = 0
        self.consumed_epsilon = 0.0
        self.template_hashes: list[str] = []
        self.unique_templates: set[str] = set()
        self.history: list[dict] = []

    @property
    def warmup_size(self) -> int:
        return max(self.cfg.min_warmup, int(self.cfg.warmup_fraction * self.cfg.k_total))

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.cfg.total_budget - self.consumed_epsilon)

    def _empirical_distribution(self) -> np.ndarray:
        """Compute Laplace-smoothed empirical p from observed history."""
        counts = Counter(self.template_hashes)
        keys = list(counts.keys())
        raw = np.array([counts[k] for k in keys], dtype=float)
        smoothed = raw + self.cfg.pseudocount
        return smoothed / smoothed.sum()

    def _predicted_total_unique(self) -> float:
        """E[u_total] under the observed history.

        With ``estimator='plugin'`` (default) this is the occupancy estimate
        ``sum_i[1-(1-p_i)^k]`` over the Laplace-smoothed empirical distribution.
        With ``estimator='smoothed_gt'`` it is the Smoothed Good--Toulmin
        unseen-species estimator, which corrects the plug-in's systematic
        under-prediction of how many *new* templates the rest of the workload
        will bring (clamped to ``[unique seen, k_total]``). Falls back to
        ``k_total`` when too little data has been seen to fit a distribution.
        """
        if len(self.template_hashes) < 2:
            return float(self.cfg.k_total)
        if self.cfg.estimator == "smoothed_gt":
            counts = list(Counter(self.template_hashes).values())
            u = predict_smoothed_gt(counts, n=self.queries_seen, k=self.cfg.k_total)
            if u == u and u > 0:  # guard NaN / degenerate output
                seen = len(self.unique_templates)
                return float(min(max(u, seen), self.cfg.k_total))
            # otherwise fall through to the plug-in estimate
        p = self._empirical_distribution()
        return expected_unique_queries(p, self.cfg.k_total)

    def next_epsilon(self, parsed: ParsedQuery) -> float:
        """Compute the eps to allocate for the upcoming query.

        Returns 0 if budget exhausted. The caller may still consult the
        cache and avoid spending anything if the query is an exact repeat.

        Allocation rule (after warmup):
            eps_q = total_budget / predicted_total_unique
        This keeps eps_q constant across cache misses (modulo prediction
        refinement) and matches the workload-aware optimum eps_q = B / u_k
        that an offline planner would pick if u_k were known.
        """
        self.queries_seen += 1
        t_hash = template_hash(extract_template(parsed))
        self.template_hashes.append(t_hash)
        self.unique_templates.add(t_hash)

        if self.remaining_budget <= self.cfg.floor_eps:
            return 0.0

        # Warmup: per-query average so the first few releases cannot drain
        # the budget under a bad prior.
        if self.queries_seen <= self.warmup_size:
            eps_q = self.cfg.total_budget / max(self.cfg.k_total, 1)
            return max(self.cfg.floor_eps, min(eps_q, self.remaining_budget))

        # Active phase: use the offline-optimal split, eps_q = B / E[u_total].
        # As we observe more queries, E[u_total] is re-estimated, so eps_q
        # may drift slowly. The allocator caps to remaining budget so total
        # spend never exceeds B by construction.
        u_total_pred = self._predicted_total_unique()
        eps_q = self.cfg.total_budget / max(u_total_pred, 1.0)
        eps_q = max(self.cfg.floor_eps, min(eps_q, self.remaining_budget))
        return eps_q

    def note_release(self, parsed: ParsedQuery, eps_used: float):
        """Record that the middleware actually consumed eps_used for this query."""
        self.consumed_epsilon += eps_used
        self.history.append({
            "query_idx": self.queries_seen,
            "unique_so_far": len(self.unique_templates),
            "eps_used": eps_used,
            "remaining_budget": self.remaining_budget,
        })

    def summary(self) -> dict:
        return {
            "total_queries": self.queries_seen,
            "unique_templates": len(self.unique_templates),
            "consumed_epsilon": self.consumed_epsilon,
            "remaining_budget": self.remaining_budget,
            "warmup_size": self.warmup_size,
        }
