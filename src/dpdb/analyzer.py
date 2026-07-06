"""Sensitivity analysis for supported aggregate queries."""

from dataclasses import dataclass
from typing import Optional

from dpdb.config import Config
from dpdb.parser import AggregateInfo, ParsedQuery
from dpdb.join_sensitivity import compute_join_sensitivity


@dataclass
class SensitivityResult:
    func: str
    column: Optional[str]
    sensitivity: float
    notes: str = ""


class SensitivityError(Exception):
    pass


def analyze_sensitivity(
    parsed: ParsedQuery, config: Config
) -> list[SensitivityResult]:
    """Compute global sensitivity for each aggregate in the query.

    Assumes tuple-level differential privacy: adding or removing one row.
    """
    # FK-join COUNT: the privacy unit is the parent entity, so removing one entity
    # removes up to d_max joined rows -> Delta f = d_max (the public FK multiplicity),
    # not 1. Only COUNT reaches here over a join (the parser rejects SUM/AVG joins).
    if parsed.is_join:
        if len(parsed.tables) != 2:
            raise SensitivityError("A join must have exactly two tables")
        jb = config.get_join_bound(parsed.tables[0], parsed.tables[1])
        if jb is None:
            raise SensitivityError(
                f"No FK multiplicity (d_max) configured for the join "
                f"{parsed.tables[0]} <-> {parsed.tables[1]}. Add it to "
                "fk_multiplicity in config.yaml; without a public bound the join "
                "sensitivity is unbounded.")
        parent, child, d_max = jb
        results = []
        for agg in parsed.aggregates:
            if agg.func != "COUNT":  # defensive: parser already enforces this
                raise SensitivityError("Only COUNT is supported over a join")
            results.append(SensitivityResult(
                func="COUNT", column=agg.column, sensitivity=d_max,
                notes=(f"Join COUNT sensitivity = d_max = {d_max:g} "
                       f"(max {child} rows per {parent} entity; privacy unit = {parent})"),
            ))
        return results

    results = []
    for agg in parsed.aggregates:
        result = _sensitivity_for_aggregate(agg, parsed.table, config)
        results.append(result)
    return results


def analyze_join_sensitivity_elastic(
    parsed: ParsedQuery,
    mf_left: int,
    mf_right: int,
    epsilon: float,
    delta: float,
) -> list[SensitivityResult]:
    """OPTIONAL, opt-in elastic/smooth-sensitivity path for a single equi-join COUNT.

    This is a DATA-DEPENDENT alternative to the conservative public ``d_max``
    clamp returned by :func:`analyze_sensitivity`. It is NOT the default and is
    never reached by the existing middleware path; a caller must invoke it
    explicitly and supply the join-key max-frequency statistics
    (``mf_left``/``mf_right``, e.g. from
    :func:`dpdb.join_sensitivity.max_frequency` over the real tables).

    Honest caveat: the smooth-sensitivity noise scale this enables is
    (eps, delta)-DP, NOT pure (eps, 0)-DP -- unlike the d_max clamp. The
    ``sensitivity`` field reports the beta-smooth sensitivity S*_beta; the exact
    local sensitivity and ES(0) are recorded in ``notes``.

    Only a single 2-table inner equi-join COUNT is supported (the parser's
    envelope); anything else raises SensitivityError.
    """
    if not parsed.is_join or len(parsed.tables) != 2:
        raise SensitivityError(
            "Elastic join sensitivity supports exactly one 2-table equi-join")
    if len(parsed.aggregates) != 1 or parsed.aggregates[0].func != "COUNT":
        raise SensitivityError(
            "Elastic join sensitivity supports a single COUNT only")
    js = compute_join_sensitivity(
        mf_left, mf_right, epsilon, delta,
        left_table=parsed.tables[0], right_table=parsed.tables[1],
    )
    return [SensitivityResult(
        func="COUNT",
        column=parsed.aggregates[0].column,
        sensitivity=js.smooth_sensitivity,
        notes=(
            f"Elastic/smooth join sensitivity (FLEX, PVLDB 2018): "
            f"LS0=max(mf_left={js.mf_left}, mf_right={js.mf_right})="
            f"{js.local_sensitivity:g}; S*_beta={js.smooth_sensitivity:.4f} at "
            f"beta={js.beta:.4g}, k*={js.k_star}. (eps,delta)-DP via NRS smooth "
            f"sensitivity (NOT pure eps-DP)."
        ),
    )]


def _sensitivity_for_aggregate(
    agg: AggregateInfo, table: str, config: Config
) -> SensitivityResult:
    if agg.func == "COUNT":
        # Adding/removing one row changes COUNT by at most 1
        return SensitivityResult(
            func="COUNT",
            column=agg.column,
            sensitivity=1.0,
            notes="Unit sensitivity under tuple-level DP",
        )

    if agg.func == "SUM":
        if agg.column is None:
            raise SensitivityError("SUM requires a column (not *)")
        bound = config.get_bound(table, agg.column)
        if bound is None:
            raise SensitivityError(
                f"No upper bound configured for {table}.{agg.column}. "
                "Add it to column_bounds in config.yaml."
            )
        # Adding/removing one row changes SUM by at most |bound|
        return SensitivityResult(
            func="SUM",
            column=agg.column,
            sensitivity=bound,
            notes=f"Bounded contribution: |{agg.column}| <= {bound}",
        )

    if agg.func == "AVG":
        if agg.column is None:
            raise SensitivityError("AVG requires a column (not *)")
        bound = config.get_bound(table, agg.column)
        if bound is None:
            raise SensitivityError(
                f"No upper bound configured for {table}.{agg.column}. "
                "Add it to column_bounds in config.yaml."
            )
        # AVG is released as a single conservative Laplace mechanism with
        # sensitivity = column bound B_c (worst case: group of size one). There is
        # NO SUM/COUNT decomposition; that amortization is future work (report
        # Future Work). Using a fixed B_c avoids the implicit n-leak of the
        # empirical-mean mechanism Lap(B_c/(n*eps)), since the group size n is private.
        return SensitivityResult(
            func="AVG",
            column=agg.column,
            sensitivity=bound,
            notes=f"Single Laplace release; worst-case bound B_c = {bound} (group of size 1)",
        )

    raise SensitivityError(f"Unsupported aggregate: {agg.func}")
