"""Composition-agnostic forecast: the budget lever the model predicts is the
PAID-RELEASE COUNT u_k, and that count is invariant to the composition theorem.

The standard DP-reviewer objection is "you used basic (linear) composition, so the
savings are inflated." This script answers it. Exact repeats are served from the
noisy cache as post-processing, so they cost ZERO budget under ANY accounting; the
number of releases that actually spend budget is therefore u_k (distinct), whether
the custodian accounts with pure epsilon-DP, with advanced (eps,delta) composition,
or with zCDP (rho). The forecast predicts u_k before execution, so it forecasts the
budget lever for all three.

What DOES change between accounting methods is how a paid-release count maps to a
total budget:
  pure eps-DP (Laplace)      : total = n * eps_q                  (linear)
  zCDP rho (Gaussian)        : total rho = n * rho_q              (linear)
  advanced (eps,delta)       : total ~ sqrt(2 n ln(1/delta)) eps_q + n eps_q(e^{eps_q}-1)  (sub-linear)

So in the linear currencies the saving is EXACTLY the model's S(k)=1-u_k/k; under
sub-linear advanced composition it is 1 - adv(u_k)/adv(k) ~ 1 - sqrt(u_k/k), still
substantial. The point: the model forecasts u_k; each composition theorem turns the
same u_k into its own budget number. Deterministic seeds.

Run: python experiments/composition_invariance.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dpdb.model import budget_savings_ratio, expected_unique_queries, zipf_distribution

M = 50
K = 100
EPS_Q = 0.5
DELTA = 1e-6
TRIALS = 200
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def adv_compose(eps_q: float, n: int, delta: float) -> float:
    """Dwork-Rothblum-Vadhan advanced composition: n eps_q-DP releases are
    (eps,delta)-DP with this eps (the standard bound)."""
    if n == 0:
        return 0.0
    return math.sqrt(2 * n * math.log(1 / delta)) * eps_q + n * eps_q * (math.exp(eps_q) - 1)


def zcdp_total_eps(rho_q: float, n: int, delta: float) -> float:
    """n Gaussian releases of rho_q-zCDP each compose to n*rho_q-zCDP, converted
    to (eps,delta): eps = rho + 2 sqrt(rho ln(1/delta))."""
    rho = n * rho_q
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def main():
    rho_q = EPS_Q ** 2 / 2.0   # a comparable per-release zCDP cost
    rows = []
    for alpha in ALPHAS:
        p = zipf_distribution(M, alpha)
        forecast_uk = expected_unique_queries(p, K)
        s_model = budget_savings_ratio(p, K)             # 1 - E[u_k]/k

        uks = []
        for t in range(TRIALS):
            rng = np.random.default_rng(9001 + 13 * t + int(alpha * 100))
            stream = rng.choice(M, size=K, p=p)
            uks.append(len(set(stream.tolist())))
        u_k = float(np.mean(uks))                        # realized paid-release count

        # savings = 1 - workload-aware / naive, in each accounting currency
        s_pure = 1 - (u_k * EPS_Q) / (K * EPS_Q)                       # linear -> = S(k)
        s_zcdp_rho = 1 - (u_k * rho_q) / (K * rho_q)                   # linear -> = S(k)
        s_adv = 1 - adv_compose(EPS_Q, round(u_k), DELTA) / adv_compose(EPS_Q, K, DELTA)
        s_zcdp_eps = 1 - zcdp_total_eps(rho_q, round(u_k), DELTA) / zcdp_total_eps(rho_q, K, DELTA)

        rows.append(dict(alpha=alpha, forecast_uk=forecast_uk, realized_uk=u_k,
                         S_model=s_model, save_pure=s_pure, save_zcdp_rho=s_zcdp_rho,
                         save_advanced=s_adv, save_zcdp_eps=s_zcdp_eps))

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results" / "composition"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "composition_invariance.csv", index=False)

    print(f"=== Composition-agnostic forecast: m={M}, k={K}, eps_q={EPS_Q}, delta={DELTA} ===")
    print(f"    paid-release count = u_k under EVERY accounting; the forecast predicts it.\n")
    show = df.copy()
    for c in ("S_model", "save_pure", "save_zcdp_rho", "save_advanced", "save_zcdp_eps"):
        show[c] = (show[c] * 100).round(1)
    show = show.rename(columns={"S_model": "S(k)% model", "save_pure": "pure-eps%",
                                "save_zcdp_rho": "zCDP-rho%", "save_advanced": "adv(e,d)%",
                                "save_zcdp_eps": "zCDP->eps%"})
    print(show[["alpha", "forecast_uk", "realized_uk", "S(k)% model", "pure-eps%",
                "zCDP-rho%", "adv(e,d)%", "zCDP->eps%"]].round(2).to_string(index=False))

    print("\n=== Headline ===")
    print(f"  Forecast predicts the paid-release count u_k (within "
          f"{(df.forecast_uk - df.realized_uk).abs().max():.2f} of realized).")
    print(f"  In the LINEAR currencies (pure-eps, zCDP-rho) the saving EQUALS the model's "
          f"S(k) exactly\n    (max gap {((df.save_pure - df.S_model).abs().max())*100:.3f} pp).")
    print(f"  Under SUB-LINEAR advanced (eps,delta) composition the saving is smaller but "
          f"still\n    {df.save_advanced.min()*100:.0f}-{df.save_advanced.max()*100:.0f}% "
          f"(= 1 - adv(u_k)/adv(k)); the SAME u_k drives it.")
    print(f"  => the forecast is composition-agnostic: it predicts u_k; each theorem prices it.")
    print(f"\n  Wrote {out / 'composition_invariance.csv'} ({len(df)} rows).")


if __name__ == "__main__":
    main()
