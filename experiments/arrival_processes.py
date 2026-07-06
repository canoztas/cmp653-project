"""Arrival-process stress battery: how the i.i.d. occupancy forecast behaves
under correlated and adversarial query streams, and why the B/m allocator is
unconditionally safe regardless.

The paper's robustness section only measures POSITIVE correlation (a sticky
Markov process), where the i.i.d. forecast over-predicts u_k -> the safe
direction. It then ASSERTS, without measuring, that negative correlation or an
adaptive analyst would push u_k ABOVE i.i.d. and make the forecast optimistic.
This script measures all of them on one footing and checks the only guarantee
that has to hold unconditionally:

    u_k <= min(k, m)  for ANY arrival process,

so the safe allocator eps_q = B/m spends at most (B/m)*u_k <= B and NEVER rejects
-- even against an adversary. The i.i.d. forecast E[u_k] is an average-case
planning number; the B/m bound is the worst-case backstop.

Processes (same Zipf(alpha) stationary marginal where one applies):
  iid              : baseline, forecast is exact in expectation.
  sticky(s)        : repeat previous template w.p. s, else draw fresh (POSITIVE
                     correlation; clusters repeats -> u_k DOWN -> forecast OVER).
  antisticky(a)    : w.p. a force a template DIFFERENT from the previous, else
                     draw fresh (NEGATIVE correlation -> u_k UP -> forecast UNDER).
  without_replace  : draw without replacement until the pool is exhausted, then
                     reshuffle (maximally spreads mass -> u_k = min(k,m) fast).
  adaptive_newness : worst case -- the analyst always issues an UNSEEN template
                     while one remains (adversarial to the distinct count being
                     small) -> u_k = min(k,m) exactly.

Deterministic seeds. Run: python experiments/arrival_processes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import expected_unique_queries, zipf_distribution

M = 50
K = 100
ALPHA = 1.2
TRIALS = 300
B_TOTAL = 10.0


def _iid(rng, p, k):
    return rng.choice(len(p), size=k, p=p)


def _sticky(rng, p, k, s):
    out = np.empty(k, dtype=int)
    out[0] = rng.choice(len(p), p=p)
    for t in range(1, k):
        out[t] = out[t - 1] if rng.random() < s else rng.choice(len(p), p=p)
    return out


def _antisticky(rng, p, k, a):
    m = len(p)
    out = np.empty(k, dtype=int)
    out[0] = rng.choice(m, p=p)
    for t in range(1, k):
        if rng.random() < a:
            # force a different template, renormalising the marginal over the rest
            mask = np.ones(m, dtype=bool)
            mask[out[t - 1]] = False
            q = p * mask
            q = q / q.sum()
            out[t] = rng.choice(m, p=q)
        else:
            out[t] = rng.choice(m, p=p)
    return out


def _without_replace(rng, p, k):
    m = len(p)
    out = np.empty(k, dtype=int)
    pos = 0
    while pos < k:
        # weighted permutation of the whole pool, then take it in order
        perm = rng.choice(m, size=m, replace=False, p=p)
        take = min(m, k - pos)
        out[pos:pos + take] = perm[:take]
        pos += take
    return out


def _adaptive_newness(rng, p, k):
    """Adversary maximising distinct count: emit a fresh template each step until
    all m are used, then anything. u_k = min(k, m) by construction."""
    m = len(p)
    order = rng.permutation(m)
    out = np.empty(k, dtype=int)
    for t in range(k):
        out[t] = order[t] if t < m else order[t % m]
    return out


def _distinct(seq):
    return len(set(seq.tolist()))


def main():
    p = zipf_distribution(M, ALPHA)
    forecast = expected_unique_queries(p, K)        # i.i.d. average-case forecast

    procs = {
        "iid": lambda rng: _iid(rng, p, K),
        "sticky(0.3)": lambda rng: _sticky(rng, p, K, 0.3),
        "sticky(0.6)": lambda rng: _sticky(rng, p, K, 0.6),
        "sticky(0.9)": lambda rng: _sticky(rng, p, K, 0.9),
        "antisticky(0.6)": lambda rng: _antisticky(rng, p, K, 0.6),
        "antisticky(0.9)": lambda rng: _antisticky(rng, p, K, 0.9),
        "without_replace": lambda rng: _without_replace(rng, p, K),
        "adaptive_newness": lambda rng: _adaptive_newness(rng, p, K),
    }

    print(f"=== Arrival-process stress: Zipf({ALPHA}), m={M}, k={K}, {TRIALS} trials ===")
    print(f"    i.i.d. forecast E[u_k] = {forecast:.2f};  hard cap min(k,m) = {min(K, M)}\n")
    print(f"  {'process':>17} | {'mean u_k':>8} | {'max u_k':>7} | {'forecast/u_k':>12} | "
          f"{'<= m?':>5} | direction")
    rows = []
    for name, gen in procs.items():
        uks = []
        for t in range(TRIALS):
            rng = np.random.default_rng(31337 + 7 * t)
            uks.append(_distinct(gen(rng)))
        uks = np.array(uks)
        mean_uk, max_uk = float(uks.mean()), int(uks.max())
        ratio = forecast / mean_uk
        within = max_uk <= M
        if abs(mean_uk - forecast) < 0.5:
            direction = "exact (i.i.d.)"
        elif forecast > mean_uk:
            direction = "OVER -> safe"
        else:
            direction = "UNDER -> optimistic"
        # worst-case spend under the B/m safe allocator
        eps_q = B_TOTAL / M
        spend = eps_q * max_uk
        rows.append(dict(process=name, mean_uk=mean_uk, max_uk=max_uk,
                         forecast=forecast, ratio=ratio, within_m=within,
                         bm_spend=spend, bm_cap=B_TOTAL))
        print(f"  {name:>17} | {mean_uk:8.2f} | {max_uk:7d} | {ratio:11.2f}x | "
              f"{str(within):>5} | {direction}")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "arrivals"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "arrival_processes.csv", index=False)

    over = df[df.process.str.startswith("sticky")]
    under = df[df.process.str.startswith(("antisticky", "without", "adaptive"))]
    print("\n=== Headline ===")
    print(f"  Positive correlation (sticky): forecast OVER-predicts u_k by up to "
          f"{over.ratio.max():.2f}x -> conservative (matches the paper).")
    print(f"  Negative / adversarial (antisticky, without-replace, adaptive): forecast "
          f"UNDER-predicts (optimistic), realized u_k up to {int(under.max_uk.max())}.")
    print(f"  BUT u_k <= m = {M} in EVERY process and trial: "
          f"{bool(df.within_m.all())}.")
    print(f"  So the B/m safe allocator (eps_q={B_TOTAL/M:.2f}) spends at most "
          f"{df.bm_spend.max():.2f} <= B={B_TOTAL} and NEVER rejects -- even under the "
          f"adaptive adversary.")
    print(f"\n  Wrote {out / 'arrival_processes.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
