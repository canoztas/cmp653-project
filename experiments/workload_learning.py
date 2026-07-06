"""Workload Learning Layer (S2): does a drift-aware streaming estimator forecast
the distinct-query horizon better than a static full-history plug-in AFTER the
workload distribution shifts mid-stream?

Setup (non-stationary synthetic workload)
-----------------------------------------
There are two disjoint template sets, A and B, each of size ``M_SET``. The first
half of the stream is drawn Zipf(alpha) over set A; the second half is drawn
Zipf(alpha) over set B (a hard regime switch at the midpoint). The marginal mix
of *which* templates are hot therefore changes completely halfway through.

Two forecasters watch the stream one query at a time:

  * ``static``      -- plug-in occupancy over the FULL raw history (never
                       forgets). After the shift its history is half stale set-A
                       mass, so it under-counts how many *fresh* distinct set-B
                       templates the next horizon will reveal.
  * ``drift-aware`` -- :class:`WorkloadLearner` with EWMA forgetting; old set-A
                       mass decays away so the estimate re-centres on set B.

Ground truth
------------
At a forecast point we ask: over the NEXT ``HORIZON`` queries, how many distinct
templates will actually appear? We answer it by drawing those future queries
from the *currently active* generating distribution (the same one the stream is
using at that point) over many Monte-Carlo trials and counting distinct
templates -- the realized ``u_k``. Each forecaster's error is |forecast -
realized|.

We report the post-shift forecast error of each method (averaged over forecast
points strictly after the shift), plus the pre-shift error as a control.

Deterministic seeds (np.random.default_rng). Run:
    python experiments/workload_learning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import zipf_distribution
from dpdb.workload_learning import StaticPluginBaseline, WorkloadLearner

# --- workload geometry ---
M_SET = 40          # templates per regime (set A, set B disjoint)
ALPHA = 1.1         # Zipf shape
N_STREAM = 2000     # total queries in the stream
SHIFT_AT = N_STREAM // 2
HORIZON = 100       # forecast horizon (distinct templates over next HORIZON queries)

# --- forecast probe points (query index at which we forecast) ---
PROBES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]

# --- Monte-Carlo settings for the realized-u_k ground truth ---
GT_TRIALS = 400
SEED = 20260626


def make_stream(rng: np.random.Generator) -> np.ndarray:
    """Build the non-stationary stream of template ids.

    Set A ids are 0..M_SET-1, set B ids are M_SET..2*M_SET-1. First half Zipf
    over A, second half Zipf over B.
    """
    p = zipf_distribution(M_SET, ALPHA)
    a = rng.choice(M_SET, size=SHIFT_AT, p=p)
    b = rng.choice(M_SET, size=N_STREAM - SHIFT_AT, p=p) + M_SET
    return np.concatenate([a, b])


def active_distribution(index: int) -> tuple[np.ndarray, int]:
    """The generating distribution active at stream position ``index``.

    Returns (probability vector, id_offset). Probabilities are Zipf(alpha) over
    the active set; id_offset shifts ids into the right disjoint block.
    """
    p = zipf_distribution(M_SET, ALPHA)
    offset = 0 if index < SHIFT_AT else M_SET
    return p, offset


def realized_distinct(index: int, horizon: int, gt_rng: np.random.Generator) -> float:
    """Monte-Carlo ground truth: mean distinct templates over the next
    ``horizon`` queries drawn from the distribution active at ``index``.

    If the horizon straddles the shift, future draws switch generating
    distributions at the shift point, matching the real stream.
    """
    p, offset = active_distribution(index)
    distinct_counts = []
    for _ in range(GT_TRIALS):
        seen: set[int] = set()
        for j in range(horizon):
            pos = index + j
            pp, off = active_distribution(pos)
            tid = int(gt_rng.choice(M_SET, p=pp)) + off
            seen.add(tid)
        distinct_counts.append(len(seen))
    return float(np.mean(distinct_counts))


def main() -> None:
    rng = np.random.default_rng(SEED)
    gt_rng = np.random.default_rng(SEED + 1)

    stream = make_stream(rng)

    learner = WorkloadLearner(mode="ewma", decay=0.99)
    static = StaticPluginBaseline()

    rows = []
    probe_set = set(PROBES)
    for i, tid in enumerate(stream):
        learner.update(int(tid))
        static.update(int(tid))
        idx = i + 1  # number of queries seen so far / next position
        if idx in probe_set:
            truth = realized_distinct(idx, HORIZON, gt_rng)
            fa = learner.forecast(HORIZON)
            drift_pred = fa.plugin_uk
            drift_sgt = fa.smoothed_gt_uk
            static_pred = static.forecast_distinct(HORIZON)
            rows.append(
                dict(
                    probe=idx,
                    post_shift=idx > SHIFT_AT,
                    realized=truth,
                    drift_pred=drift_pred,
                    drift_sgt=drift_sgt,
                    static_pred=static_pred,
                    drift_err=abs(drift_pred - truth),
                    drift_sgt_err=abs(drift_sgt - truth),
                    static_err=abs(static_pred - truth),
                    drift_active=fa.active_templates,
                    static_distinct=static.distinct(),
                )
            )

    df = pd.DataFrame(rows)

    print(f"=== Workload Learning Layer: non-stationary stream "
          f"(Zipf({ALPHA}) over set A -> set B at query {SHIFT_AT}) ===")
    print(f"    {M_SET} templates/regime, horizon={HORIZON} queries, "
          f"{GT_TRIALS} MC trials/probe, EWMA decay=0.99\n")
    print(f"  {'probe':>5} | {'phase':>5} | {'realized':>8} | "
          f"{'drift_pi':>8} | {'drift_sgt':>9} | {'static':>7} | "
          f"{'pi_err':>6} | {'sgt_err':>7} | {'static_err':>10}")
    for _, r in df.iterrows():
        phase = "POST" if r.post_shift else "pre"
        print(f"  {int(r.probe):>5} | {phase:>5} | {r.realized:8.2f} | "
              f"{r.drift_pred:8.2f} | {r.drift_sgt:9.2f} | {r.static_pred:7.2f} | "
              f"{r.drift_err:6.2f} | {r.drift_sgt_err:7.2f} | {r.static_err:10.2f}")

    pre = df[~df.post_shift]
    post = df[df.post_shift]
    drift_post = post.drift_err.mean()
    drift_sgt_post = post.drift_sgt_err.mean()
    static_post = post.static_err.mean()

    out = Path(__file__).parent.parent / "results" / "workload_learning"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "workload_learning.csv", index=False)

    print("\n=== Headline ===")
    print(f"  Pre-shift mean forecast error : drift-aware(plugin) {pre.drift_err.mean():.2f} "
          f"vs static {pre.static_err.mean():.2f} (static already drifts up as its support grows).")
    print(f"  POST-shift mean forecast error: drift-aware plugin {drift_post:.2f}, "
          f"drift-aware smoothed-GT {drift_sgt_post:.2f}, static {static_post:.2f}.")
    print(f"  -> Drift-aware plug-in is {static_post / max(drift_post, 1e-9):.2f}x more accurate "
          f"than static post-shift; smoothed-GT is {static_post / max(drift_sgt_post, 1e-9):.2f}x.")
    print(f"  The static full-history plug-in keeps {int(post.static_distinct.iloc[-1])} "
          f"stale+fresh templates in its support and over-predicts the post-shift\n"
          f"  distinct-query horizon ({post.static_pred.mean():.0f} vs realized "
          f"{post.realized.mean():.0f}); the EWMA learner forgets set A and re-centres on set B.")
    print("\n  LIMIT: this layer learns the forecast (p_i, active count, E[u_k]); it does")
    print("  NOT learn m's hard cap. B/m budget safety still relies on the public m.")
    print(f"\n  Wrote {out / 'workload_learning.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
