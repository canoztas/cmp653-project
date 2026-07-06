"""Out-of-distribution / model-misspecification stress for the occupancy forecast
E[u_k]=sum_i (1-(1-p_i)^k).

WHY THIS EXISTS. The headline synthetic grid samples the workload from the SAME
Zipf marginal that is fed to the closed form, so it is a Monte-Carlo
self-consistency check (we label it as such): plug in the TRUE {p_i} and the
forecast matches by construction. The honest objection is that a custodian never
knows the true {p_i}; they ASSUME one (a Zipf prior, a default skew) and forecast
before execution. This script asks the real question: how well does a
*misspecified* forecast -- p_hat != q_true -- predict realized u_k?

Two mismatch families, both with the forecast distribution p_hat DIFFERENT from
the generating distribution q_true (so this is genuinely out-of-distribution, not
a tautology):

  (A) SKEW misspecification. Custodian assumes Zipf(alpha_hat); the workload is
      really Zipf(alpha_true). Sweep alpha_true. Because E[u_k] is monotone
      DECREASING in skew, assuming LESS skew than reality (alpha_hat < alpha_true)
      OVER-forecasts u_k -> conservative -> budget-safe; assuming MORE skew
      UNDER-forecasts -> optimistic -> unsafe. So there is a safe direction:
      pick a less-skewed (flatter) prior when the true skew is uncertain.

  (B) STRUCTURAL misspecification. The workload is not Zipf at all (uniform,
      lognormal-ish, heavy Pareto-ish); the custodian still forecasts with a
      Zipf(alpha_hat) prior. Measures error under a wrong FAMILY, not just a
      wrong parameter.

The oracle forecast E_{q_true}[u_k] is printed alongside as a ~0-error sanity
anchor (the formula IS exact for the true marginal). Deterministic seeds.
Run: python experiments/misspecification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import expected_unique_queries, uniform_distribution, zipf_distribution

M = 64
KS = (100, 500)
TRIALS = 60
ALPHA_HATS = (0.5, 1.0, 1.5)          # what the custodian ASSUMES
ALPHA_TRUES = (0.0, 0.5, 1.0, 1.5, 2.0)  # what the workload REALLY is


def _lognormalish(m, sigma=1.0):
    x = np.exp(sigma * np.arange(m) / m)[::-1]
    return x / x.sum()


def _paretoish(m, beta=2.0):
    x = (1.0 + np.arange(m)) ** (-beta)
    return x / x.sum()


def _realized_uk(q, k, n_trials):
    """Monte-Carlo mean distinct count of a length-k i.i.d. draw from q."""
    vals = []
    for t in range(n_trials):
        rng = np.random.default_rng(20260621 + 131 * k + t)
        vals.append(len(set(rng.choice(len(q), size=k, p=q).tolist())))
    return float(np.mean(vals))


def _direction(forecast, realized):
    # occupancy = expected privacy spend proxy: over-forecast never under-spends
    if abs(forecast - realized) < 1e-9:
        return "exact"
    return "OVER -> SAFE" if forecast > realized else "UNDER -> unsafe"


def main():
    rows = []

    print("=== (A) SKEW misspecification: assume Zipf(alpha_hat), truth is Zipf(alpha_true) ===")
    print("    (oracle = forecast using the TRUE marginal; ~0 error by construction)\n")
    for k in KS:
        print(f"--- k={k}, m={M} ---")
        header = (f"  {'assumed a_hat':>13} | {'true a_true':>11} | {'forecast':>9} | "
                  f"{'realized':>9} | {'oracle':>7} | {'rel.err':>8} | direction")
        print(header)
        for ah in ALPHA_HATS:
            p_hat = zipf_distribution(M, ah)
            f_mis = expected_unique_queries(p_hat, k)
            for at in ALPHA_TRUES:
                q = zipf_distribution(M, at)
                realized = _realized_uk(q, k, TRIALS)
                oracle = expected_unique_queries(q, k)
                rel = abs(f_mis - realized) / realized * 100.0
                rows.append(dict(family="skew", k=k, alpha_hat=ah, alpha_true=at,
                                 forecast=f_mis, realized=realized, oracle=oracle,
                                 rel_err_pct=rel, direction=_direction(f_mis, realized)))
                print(f"  {ah:13.1f} | {at:11.1f} | {f_mis:9.2f} | {realized:9.2f} | "
                      f"{oracle:7.2f} | {rel:7.2f}% | {_direction(f_mis, realized)}")
        print()

    print("=== (B) STRUCTURAL misspecification: assume Zipf(1.0), truth is a different FAMILY ===\n")
    families = {
        "uniform": uniform_distribution(M),
        "lognormal-ish": _lognormalish(M),
        "pareto-ish(b=2)": _paretoish(M, 2.0),
        "zipf(1.0) [match]": zipf_distribution(M, 1.0),
    }
    p_hat = zipf_distribution(M, 1.0)
    for k in KS:
        print(f"--- k={k}, assumed Zipf(1.0) ---")
        print(f"  {'true family':>18} | {'forecast':>9} | {'realized':>9} | {'rel.err':>8} | direction")
        for name, q in families.items():
            f_mis = expected_unique_queries(p_hat, k)
            realized = _realized_uk(q, k, TRIALS)
            rel = abs(f_mis - realized) / realized * 100.0
            rows.append(dict(family="structural", k=k, alpha_hat=1.0, alpha_true=name,
                             forecast=f_mis, realized=realized, oracle=expected_unique_queries(q, k),
                             rel_err_pct=rel, direction=_direction(f_mis, realized)))
            print(f"  {name:>18} | {f_mis:9.2f} | {realized:9.2f} | {rel:7.2f}% | "
                  f"{_direction(f_mis, realized)}")
        print()

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "ood"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "misspecification.csv", index=False)

    # --- Headline ---
    skew = df[df.family == "skew"]
    near = skew[(skew.alpha_hat - skew.alpha_true).abs() <= 0.5]
    safe = skew[skew.alpha_hat <= skew.alpha_true]   # assume <= true skew -> over-forecast
    print("=== Headline ===")
    print(f"  Oracle (true marginal) forecast error: mean {abs(skew.forecast*0).mean():.2f}% by design; "
          f"max |oracle-realized|/realized = "
          f"{(skew.oracle - skew.realized).abs().div(skew.realized).max()*100:.2f}% (Monte-Carlo).")
    print(f"  Misspecified forecast, |a_hat - a_true| <= 0.5: mean rel.err "
          f"{near.rel_err_pct.mean():.1f}% (max {near.rel_err_pct.max():.1f}%).")
    print(f"  Safe-direction rows (assume <= true skew): "
          f"{(safe.direction == 'OVER -> SAFE').mean()*100:.0f}% over-forecast (budget-conservative).")
    print(f"  Worst structural mismatch (Zipf prior vs non-Zipf truth): "
          f"{df[df.family=='structural'].rel_err_pct.max():.1f}% rel.err.")
    print(f"\n  Wrote {out / 'misspecification.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
