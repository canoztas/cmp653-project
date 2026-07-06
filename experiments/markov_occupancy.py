"""Markov (sticky) occupancy: a closed-form E[u_k] for bursty arrivals.

The i.i.d. occupancy E[u_k]=sum_i(1-(1-p_i)^k) assumes independent draws. Real
workloads are bursty: a template, once issued, tends to repeat. We model this
with the sticky Markov process (X_1~p; X_t=X_{t-1} w.p. s, else fresh ~p) and
derive its occupancy in closed form, EXTENDING the model from i.i.d. to a
Markovian arrival process (not merely stress-testing the i.i.d. forecast).

Closed form (Proposition, validated here):
    E[u_k] = sum_i [ 1 - (1-p_i) * (s + (1-s)(1-p_i))^{k-1} ].

Derivation: the distinct templates visited equal the distinct values among the
FRESH draws (a repeat re-emits the current template); the number of fresh draws
is F = 1 + Binomial(k-1, 1-s); condition on F and use the Binomial PGF.

This script (i) confirms the closed form matches a 4000-trial simulation across
s and (m,k,alpha), and (ii) shows the i.i.d. forecast equals the s=0 case and
over-predicts for s>0 -- so on bursty streams it is a safe (conservative) upper
bound, now quantified exactly rather than only bounded by u_k<=m.

Deterministic seeds. Run: python experiments/markov_occupancy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import (expected_unique_queries, expected_unique_queries_sticky,
                        expected_unique_queries_markov, zipf_distribution)

TRIALS = 4000
CELLS = [(20, 100, 1.0), (50, 100, 1.2), (50, 200, 0.8), (100, 150, 1.0)]
STICKY = (0.0, 0.3, 0.6, 0.9)


def _draw_sticky(rng, p, k, s):
    out = np.empty(k, dtype=int)
    out[0] = rng.choice(len(p), p=p)
    for t in range(1, k):
        out[t] = out[t - 1] if rng.random() < s else rng.choice(len(p), p=p)
    return out


def main():
    rows = []
    print("=== Sticky-Markov occupancy: closed form vs simulation ===\n")
    print(f"  {'m':>3} {'k':>4} {'a':>4} {'s':>4} | {'closed':>7} {'sim':>7} {'|err|':>6} | "
          f"{'iid':>6} | direction")
    for (m, k, alpha) in CELLS:
        p = zipf_distribution(m, alpha)
        iid = expected_unique_queries(p, k)
        for s in STICKY:
            closed = expected_unique_queries_sticky(p, k, s)
            sim = float(np.mean([
                len(set(_draw_sticky(np.random.default_rng(7000 + 11 * t), p, k, s).tolist()))
                for t in range(TRIALS)]))
            err = abs(closed - sim)
            direction = ("exact (=i.i.d.)" if s == 0 else
                         "i.i.d. OVER-predicts -> safe" if iid > sim else "?")
            rows.append(dict(m=m, k=k, alpha=alpha, s=s, closed=closed, sim=sim,
                             abs_err=err, iid=iid))
            print(f"  {m:>3} {k:>4} {alpha:>4.1f} {s:>4.1f} | {closed:7.3f} {sim:7.3f} "
                  f"{err:6.3f} | {iid:6.2f} | {direction}")
        print()

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "markov"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "markov_occupancy.csv", index=False)

    # General-Markov occupancy: arbitrary transition matrix vs simulation, and
    # the i.i.d./sticky forms recovered as special cases (machine precision).
    print("=== General-Markov occupancy: closed form vs simulation (random chains) ===\n")
    print(f"  {'m':>3} {'k':>4} | {'closed':>7} {'sim':>7} {'|err|':>6}")
    gen_err = []
    for (m, k) in ((6, 60), (8, 100), (10, 150)):
        rng0 = np.random.default_rng(2024 + m)
        Mx = rng0.random((m, m)) + 0.05
        P = Mx / Mx.sum(1, keepdims=True)            # random row-stochastic (non-sticky)
        nu = rng0.random(m); nu /= nu.sum()
        closed = expected_unique_queries_markov(P, nu, k)
        sims = []
        for t in range(TRIALS):
            r = np.random.default_rng(8000 + 13 * t)
            x = r.choice(m, p=nu); seen = {x}
            for _ in range(k - 1):
                x = r.choice(m, p=P[x]); seen.add(x)
            sims.append(len(seen))
        sim = float(np.mean(sims)); gen_err.append(abs(closed - sim))
        print(f"  {m:>3} {k:>4} | {closed:7.3f} {sim:7.3f} {abs(closed - sim):6.3f}")
    # special-case recovery
    p = zipf_distribution(8, 1.0)
    iid_gap = abs(expected_unique_queries_markov(np.tile(p, (8, 1)), p, 80)
                  - expected_unique_queries(p, 80))
    s = 0.6
    Pst = (1 - s) * np.tile(p, (8, 1)) + s * np.eye(8)
    sticky_gap = abs(expected_unique_queries_markov(Pst, p, 80)
                     - expected_unique_queries_sticky(p, 80, s))
    print(f"\n  Recovers i.i.d. to {iid_gap:.2e} and sticky to {sticky_gap:.2e} "
          f"(machine precision); general chains within max |err| {max(gen_err):.3f}.\n")

    print("=== Headline ===")
    print(f"  Closed form matches simulation to a max |error| of {df.abs_err.max():.3f} "
          f"(Monte-Carlo, {TRIALS} trials) across all {len(df)} cells.")
    s0 = df[df.s == 0.0]
    print(f"  At s=0 the sticky form equals the i.i.d. forecast (max gap "
          f"{(s0.closed - s0.iid).abs().max():.4f}); for s>0 it is strictly smaller, so the "
          f"i.i.d.\n    forecast over-predicts bursty u_k -- the burstiness gap is now an "
          f"exact quantity, not just bounded by u_k<=m.")
    print(f"\n  Wrote {out / 'markov_occupancy.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
