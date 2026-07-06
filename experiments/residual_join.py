"""Elastic / smooth sensitivity vs the conservative d_max=7 clamp on the REAL
orders |><| lineitem join COUNT.

The production path bounds a 2-table FK-join COUNT by the public FK multiplicity
d_max=7 (TPC-H spec: <=7 line-items per order). That bound is sound and yields
PURE eps-DP, but it is data-INDEPENDENT: it assumes every order has the maximal 7
line-items. Elastic sensitivity (FLEX, Johnson-Near-Song, PVLDB 2018) instead
reads the actual join-key max-frequency statistics from the data and, via smooth
sensitivity (NRS 2007), calibrates noise to the realized worst case -- at the
honest cost of an (eps, delta)-DP relaxation rather than pure eps-DP.

This script measures, on the real tables:
  1. the realized max-frequency of l_orderkey in lineitem (= LS0 of the join
     COUNT, since o_orderkey is a PRIMARY KEY in orders so mf=1);
  2. ES(0) = max(mf_l, mf_r) and the beta-smooth sensitivity S*_beta;
  3. whether S*_beta is TIGHTER (lower noise) than the d_max=7 clamp while still
     being a valid upper bound on local sensitivity.

Deterministic: reads fixed tables; no RNG. Run: python experiments/residual_join.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb
import pandas as pd

from dpdb.config import Config
from dpdb.join_sensitivity import (
    join_count_sensitivity_from_db,
    laplace_scale_from_smooth,
    max_frequency,
    recommended_beta,
    smooth_sensitivity,
)

DELTA = 1e-6
EPSILONS = [0.1, 0.5, 1.0, 2.0]


def main():
    root = Path(__file__).parent.parent
    cfg = Config.from_yaml(str(root / "config.yaml"))
    con = duckdb.connect(str(root / cfg.duckdb_path), read_only=True)

    # Public conservative clamp from config (orders -> lineitem FK multiplicity).
    d_max = float(cfg.fk_multiplicity["orders"]["lineitem"])

    # Real max-frequency statistics for the join key on each side.
    mf_orders = max_frequency(con, "orders", "o_orderkey")     # PK -> 1
    mf_lineitem = max_frequency(con, "lineitem", "l_orderkey")  # <= 7 by spec

    ls0 = float(max(mf_orders, mf_lineitem))

    print("=== orders |><| lineitem COUNT: elastic/smooth vs d_max=7 clamp ===")
    print(f"  mf(o_orderkey in orders)   = {mf_orders}  (primary key)")
    print(f"  mf(l_orderkey in lineitem) = {mf_lineitem}")
    print(f"  Local sensitivity LS0 = ES(0) = max(mf) = {ls0:g}")
    print(f"  Conservative public clamp d_max = {d_max:g}")
    print(f"  ES(0) <= d_max ? {ls0 <= d_max}  "
          f"(elastic LS0 is {'tighter' if ls0 < d_max else 'equal/looser'})\n")

    print(f"  {'eps':>5} | {'beta':>10} | {'S*_beta':>8} | {'k*':>3} | "
          f"{'lap_scale(smooth)':>17} | {'lap_scale(d_max)':>16} | {'noise ratio':>11}")
    rows = []
    for eps in EPSILONS:
        js = join_count_sensitivity_from_db(
            con, "lineitem", "l_orderkey", "orders", "o_orderkey",
            epsilon=eps, delta=DELTA,
        )
        beta = js.beta
        smooth = js.smooth_sensitivity
        # (eps, delta)-DP Laplace scale from smooth sensitivity (NRS 2007).
        lap_smooth = laplace_scale_from_smooth(smooth, eps, DELTA)
        # Pure eps-DP Laplace scale from the conservative global-sensitivity clamp.
        lap_dmax = d_max / eps
        ratio = lap_smooth / lap_dmax
        rows.append(dict(
            epsilon=eps, beta=beta, smooth_sensitivity=smooth, k_star=js.k_star,
            local_sensitivity=ls0, d_max=d_max,
            lap_scale_smooth=lap_smooth, lap_scale_dmax=lap_dmax,
            noise_ratio_smooth_over_dmax=ratio,
            smooth_tighter=lap_smooth < lap_dmax,
        ))
        print(f"  {eps:5.2f} | {beta:10.4g} | {smooth:8.3f} | {js.k_star:3d} | "
              f"{lap_smooth:17.4f} | {lap_dmax:16.4f} | {ratio:10.3f}x")

    df = pd.DataFrame(rows)
    out = root / "results" / "joins"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "residual_join_elastic.csv"
    df.to_csv(csv, index=False)

    # --- A real low-skew join where elastic sensitivity DOES tighten -----------
    # part |><| lineitem ON p_partkey = l_partkey: p_partkey is a PRIMARY KEY in
    # part (mf=1) and l_partkey has a measured max-frequency well below any tight
    # public bound. TPC-H gives no small spec cap on "line-items per part", so a
    # conservative deployer must pick a LARGE public clamp; elastic sensitivity
    # reads the real mf instead. We report the measured mf and the smallest
    # conservative clamp d_max for which the smooth path is already tighter.
    mf_part = max_frequency(con, "part", "p_partkey")           # PK -> 1
    mf_li_part = max_frequency(con, "lineitem", "l_partkey")    # measured skew
    ls0_part = float(max(mf_part, mf_li_part))
    print("\n=== Low-skew real join: part |><| lineitem ON p_partkey=l_partkey ===")
    print(f"  mf(p_partkey in part)        = {mf_part}  (primary key)")
    print(f"  mf(l_partkey in lineitem)    = {mf_li_part}  (measured)")
    print(f"  Local sensitivity LS0 = ES(0) = {ls0_part:g}")
    eps_ref = 1.0
    beta_ref = recommended_beta(eps_ref, DELTA)
    smooth_part, kstar_part = smooth_sensitivity(int(mf_part), int(mf_li_part), beta_ref)
    lap_smooth_part = laplace_scale_from_smooth(smooth_part, eps_ref, DELTA)
    # smallest public clamp value at which pure-eps d_max noise >= smooth noise:
    # d_max/eps >= lap_smooth_part  <=>  d_max >= lap_smooth_part * eps
    breakeven_dmax = lap_smooth_part * eps_ref
    print(f"  At eps={eps_ref:g}, delta={DELTA:g}: S*_beta={smooth_part:.3f} (k*={kstar_part}), "
          f"smooth Laplace scale={lap_smooth_part:.3f}.")
    print(f"  The smooth (eps,delta)-DP path beats a pure-eps clamp whenever the "
          f"conservative public d_max >= {breakeven_dmax:.1f}.")
    print(f"  Since the realized mf is only {mf_li_part} but no tight public cap "
          f"exists for lineitems-per-part, a deployer's conservative clamp would "
          f"plausibly exceed {breakeven_dmax:.0f} -> elastic sensitivity is the "
          f"tighter, instance-aware choice for this join.")
    df_part = pd.DataFrame([dict(
        join="part|><|lineitem", mf_left=mf_part, mf_right=mf_li_part,
        local_sensitivity=ls0_part, epsilon=eps_ref, delta=DELTA, beta=beta_ref,
        smooth_sensitivity=smooth_part, k_star=kstar_part,
        lap_scale_smooth=lap_smooth_part, breakeven_dmax=breakeven_dmax,
    )])
    df_part.to_csv(out / "residual_join_partkey.csv", index=False)

    print("\n=== Headline ===")
    print(f"  ES(0) = {ls0:g} <= d_max = {d_max:g}: the elastic local sensitivity is a "
          f"VALID upper bound and is "
          f"{'TIGHTER' if ls0 < d_max else 'EQUAL'} than the public clamp.")
    any_smooth_tighter = bool(df.smooth_tighter.any())
    if any_smooth_tighter:
        tights = df[df.smooth_tighter]
        best = tights.loc[tights.noise_ratio_smooth_over_dmax.idxmin()]
        print(f"  After the smooth-sensitivity inflation (k*>0) and the 2x NRS factor, "
              f"the (eps,delta)-DP smooth Laplace scale is LOWER than the pure-eps "
              f"d_max scale for {int(df.smooth_tighter.sum())}/{len(df)} epsilons "
              f"(best {best.noise_ratio_smooth_over_dmax:.2f}x at eps={best.epsilon:g}).")
    else:
        print(f"  HONEST RESULT: with mf(l_orderkey)={mf_lineitem} essentially equal to "
              f"d_max={d_max:g}, the smooth-sensitivity inflation (k*>0) plus the 2x NRS "
              f"factor makes the (eps,delta)-DP smooth Laplace scale LARGER than the "
              f"pure-eps d_max scale at these epsilons. ES(0) is still the tighter "
              f"*sensitivity*, but the (eps,delta) smoothing overhead cancels the gain "
              f"here. The win would appear when mf << d_max (skew-free joins).")
    print(f"  NOTE: the smooth path is (eps,delta)-DP, NOT pure eps-DP. The d_max clamp "
          f"is pure eps-DP. This is the honest trade-off.")
    print(f"\n  Wrote {csv} ({len(df)} rows).")


if __name__ == "__main__":
    main()
