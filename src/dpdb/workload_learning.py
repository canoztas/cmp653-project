"""Online Workload Learning Layer (professor suggestion S2).

The static budget allocator estimates the template distribution ``{p_i}`` and
the distinct-template horizon ``u_k`` from a *fixed prefix* of the query stream
and never revises it. That is fine for a stationary workload, but real analyst
streams drift: the popular query mix at 9am is not the mix at 5pm, dashboards
get added and retired, ad-hoc investigations spike and fade. A prefix estimate
fitted before a shift keeps forecasting the *old* distribution forever.

This module adds a streaming estimator, :class:`WorkloadLearner`, that ingests
queries one at a time and continuously re-estimates

  * the live template-frequency distribution ``{p_i}`` (decayed / windowed),
  * the live number of *active* templates ``m_hat`` (templates with non-trivial
    recent mass), and
  * a forecast of the distinct templates / ``E[u_k]`` over the next horizon,

reusing the existing :mod:`dpdb.predictors` (plug-in occupancy and smoothed
Good-Toulmin) applied to the *current* decayed counts rather than the raw
full-history counts. Because the counts are exponentially decayed (or held in a
sliding window), templates that stop arriving fade out and templates that start
arriving fade in, so the forecast tracks the drift instead of averaging across
it.

Two decay modes:

  * ``"ewma"``   -- exponential forgetting: every ingested query multiplies all
                    counts by ``decay`` (0<decay<1) before incrementing the
                    current template. The effective window length is
                    ``1/(1-decay)``. Counts become fractional; the predictors
                    are fed *rounded* effective counts so the
                    frequency-of-frequencies statistics they need are integers.
  * ``"window"`` -- a hard sliding window of the last ``window`` queries; counts
                    are exact integer occurrences inside the window.

HONEST LIMITS
-------------
* This layer learns the *forecast* (``{p_i}``, the active count, ``E[u_k]``).
  It does NOT learn ``m``'s hard cap. The unconditional budget-safety argument
  ``u_k <= min(k, m)`` and the safe allocator ``eps_q = B/m`` still rely on a
  *public* upper bound ``m`` on the number of distinct templates the schema can
  ever produce. ``m_hat`` here is an *active-template* estimate for planning the
  average-case forecast, never a replacement for that public cap.
* The active-template count ``m_hat`` is thresholded on decayed mass, so it is a
  soft, drift-following estimate, not an exact richness estimator.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Optional

import numpy as np

# Allow ``import dpdb...`` whether or not src/ is already on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpdb.predictors import (  # noqa: E402
    predict_plugin,
    predict_smoothed_gt,
)


@dataclass
class Forecast:
    """A snapshot of the learner's current estimate.

    Attributes
    ----------
    n_seen:
        Total queries ingested so far (raw, not decayed).
    active_templates:
        ``m_hat`` -- number of templates with non-trivial recent (decayed) mass.
    distinct_recent:
        Number of distinct templates with any positive decayed count.
    plugin_uk:
        Plug-in occupancy forecast of distinct templates over a horizon of
        ``k`` *future* queries, using the current decayed distribution.
    smoothed_gt_uk:
        Smoothed Good-Toulmin forecast over the same horizon.
    horizon_k:
        The horizon (number of future queries) the forecasts refer to.
    """

    n_seen: int
    active_templates: int
    distinct_recent: int
    plugin_uk: float
    smoothed_gt_uk: float
    horizon_k: int


class WorkloadLearner:
    """Streaming, drift-aware estimator of the template workload.

    Parameters
    ----------
    mode:
        ``"ewma"`` (exponential forgetting) or ``"window"`` (hard sliding
        window). Default ``"ewma"``.
    decay:
        EWMA forgetting factor in (0, 1). Each ingested query multiplies all
        running counts by ``decay`` before the current template is incremented.
        Closer to 1 = longer memory. Only used in ``"ewma"`` mode.
    window:
        Sliding-window length (number of most-recent queries retained). Only
        used in ``"window"`` mode.
    active_threshold:
        A template counts toward ``m_hat`` (active templates) when its decayed
        share of the current total mass is at least this fraction. Default
        ``0.0`` means "any positive decayed count"; a small positive value
        (e.g. ``0.005``) excludes faded-out tails.
    """

    def __init__(
        self,
        mode: str = "ewma",
        decay: float = 0.99,
        window: int = 200,
        active_threshold: float = 0.0,
    ) -> None:
        if mode not in ("ewma", "window"):
            raise ValueError("mode must be 'ewma' or 'window'")
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        if window < 1:
            raise ValueError("window must be >= 1")
        if active_threshold < 0.0:
            raise ValueError("active_threshold must be >= 0")
        self.mode = mode
        self.decay = float(decay)
        self.window = int(window)
        self.active_threshold = float(active_threshold)

        # Decayed (or windowed) counts keyed by template id.
        self._counts: dict[Hashable, float] = {}
        # Hard-window history (only used in window mode).
        self._history: deque = deque()
        # Window integer counts (only used in window mode).
        self._win_counts: dict[Hashable, int] = {}
        self._n_seen = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def update(self, template_id: Hashable) -> None:
        """Ingest a single query identified by ``template_id``."""
        self._n_seen += 1
        if self.mode == "ewma":
            d = self.decay
            # Multiplicatively decay all existing counts, then add the new one.
            for key in self._counts:
                self._counts[key] *= d
            self._counts[template_id] = self._counts.get(template_id, 0.0) + 1.0
            # Periodically prune negligible counts to bound the dict size.
            if self._n_seen % 256 == 0:
                self._prune()
        else:  # window
            self._history.append(template_id)
            self._win_counts[template_id] = self._win_counts.get(template_id, 0) + 1
            if len(self._history) > self.window:
                old = self._history.popleft()
                self._win_counts[old] -= 1
                if self._win_counts[old] <= 0:
                    del self._win_counts[old]

    def update_many(self, template_ids: Iterable[Hashable]) -> None:
        """Ingest a batch of queries in order."""
        for t in template_ids:
            self.update(t)

    def _prune(self, eps: float = 1e-9) -> None:
        total = sum(self._counts.values())
        if total <= 0:
            return
        floor = eps * total
        self._counts = {k: v for k, v in self._counts.items() if v >= floor}

    # ------------------------------------------------------------------
    # Current estimate
    # ------------------------------------------------------------------
    @property
    def n_seen(self) -> int:
        return self._n_seen

    def _raw_counts(self) -> dict[Hashable, float]:
        if self.mode == "ewma":
            return self._counts
        return {k: float(v) for k, v in self._win_counts.items()}

    def current_distribution(self) -> np.ndarray:
        """Current decayed/windowed template distribution ``{p_i}`` as an array.

        The array is sorted by descending mass; index has no template meaning.
        Returns an empty array before any query has been seen.
        """
        counts = self._raw_counts()
        if not counts:
            return np.zeros(0, dtype=float)
        vals = np.array(sorted(counts.values(), reverse=True), dtype=float)
        s = vals.sum()
        if s <= 0:
            return np.zeros(0, dtype=float)
        return vals / s

    def _effective_counts(self) -> np.ndarray:
        """Integer-rounded effective counts for the frequency-of-frequencies
        statistics the predictors consume.

        For EWMA the running counts are fractional; the effective sample size is
        the *current total decayed mass*. We rescale the normalized distribution
        back to that mass and round, so the smoothed Good-Toulmin estimator sees
        a plausible integer histogram (phi_1, phi_2, ...). For window mode the
        counts are already integers.
        """
        counts = self._raw_counts()
        if not counts:
            return np.zeros(0, dtype=float)
        vals = np.array(list(counts.values()), dtype=float)
        if self.mode == "window":
            return vals
        # EWMA: effective sample size = total decayed mass.
        n_eff = vals.sum()
        p = vals / n_eff
        eff = np.round(p * n_eff)
        eff = eff[eff > 0]
        return eff

    def effective_n(self) -> int:
        """The effective number of samples backing the current estimate.

        For window mode this is the number of queries in the window; for EWMA it
        is the rounded total decayed mass (the steady-state value is
        ``1/(1-decay)``).
        """
        if self.mode == "window":
            return int(min(self._n_seen, self.window))
        return int(round(sum(self._counts.values())))

    def active_templates(self) -> int:
        """``m_hat``: templates whose decayed share is >= ``active_threshold``.

        With the default threshold of 0 this is just the number of distinct
        templates with positive decayed mass. NOTE: this is an *active-template*
        estimate for planning the forecast, NOT the public hard cap ``m`` the
        budget-safety bound relies on.
        """
        p = self.current_distribution()
        if p.size == 0:
            return 0
        return int(np.sum(p >= self.active_threshold))

    def distinct_recent(self) -> int:
        """Distinct templates with any positive decayed/windowed count."""
        counts = self._raw_counts()
        return int(sum(1 for v in counts.values() if v > 0))

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def forecast(self, horizon_k: int) -> Forecast:
        """Forecast distinct templates over the next ``horizon_k`` queries.

        Both forecasts use the *current* decayed distribution as the estimate of
        ``{p_i}``. The plug-in occupancy forecast is ``sum_i [1-(1-p_i)^k]`` over
        the rounded effective counts; the smoothed Good-Toulmin forecast
        extrapolates the unseen-template tail from the same effective histogram.

        ``horizon_k`` is the number of FUTURE queries (the predictors are called
        with ``n = effective_n`` and ``k = effective_n + horizon_k`` so that the
        Good-Toulmin horizon factor ``t = horizon_k / n`` is measured from the
        current effective sample).
        """
        if horizon_k < 0:
            raise ValueError("horizon_k must be >= 0")
        eff = self._effective_counts()
        n_eff = self.effective_n()
        if eff.size == 0 or n_eff <= 0:
            return Forecast(
                n_seen=self._n_seen,
                active_templates=0,
                distinct_recent=0,
                plugin_uk=0.0,
                smoothed_gt_uk=0.0,
                horizon_k=horizon_k,
            )
        k_total = n_eff + horizon_k
        plugin = predict_plugin(eff, n_eff, k_total)
        sgt = predict_smoothed_gt(eff, n_eff, k_total)
        return Forecast(
            n_seen=self._n_seen,
            active_templates=self.active_templates(),
            distinct_recent=self.distinct_recent(),
            plugin_uk=float(plugin),
            smoothed_gt_uk=float(sgt),
            horizon_k=horizon_k,
        )

    def expected_remaining_distinct(self, horizon_k: int) -> float:
        """Convenience: forecast distinct templates over the next ``horizon_k``
        queries via the plug-in occupancy estimator on the current distribution.
        """
        return self.forecast(horizon_k).plugin_uk


class StaticPluginBaseline:
    """Non-adaptive baseline: a plug-in forecast over the FULL raw history.

    It accumulates exact integer counts over every query ever seen and never
    forgets, so after a distribution shift it keeps mixing the pre-shift and
    post-shift mass. This is the estimator the drift-aware learner is compared
    against in the experiment.
    """

    def __init__(self) -> None:
        self._counts: dict[Hashable, int] = {}
        self._n_seen = 0

    def update(self, template_id: Hashable) -> None:
        self._counts[template_id] = self._counts.get(template_id, 0) + 1
        self._n_seen += 1

    def update_many(self, template_ids: Iterable[Hashable]) -> None:
        for t in template_ids:
            self.update(t)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def distinct(self) -> int:
        return len(self._counts)

    def forecast_distinct(self, horizon_k: int) -> float:
        """Plug-in occupancy forecast over the next ``horizon_k`` queries using
        the full-history counts."""
        if not self._counts:
            return 0.0
        counts = np.array(list(self._counts.values()), dtype=float)
        n = int(counts.sum())
        return float(predict_plugin(counts, n, n + horizon_k))
