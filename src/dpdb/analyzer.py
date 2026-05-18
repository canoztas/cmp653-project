"""Sensitivity analysis for supported aggregate queries."""

from dataclasses import dataclass
from typing import Optional

from dpdb.config import Config
from dpdb.parser import AggregateInfo, ParsedQuery


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
    results = []
    for agg in parsed.aggregates:
        result = _sensitivity_for_aggregate(agg, parsed.table, config)
        results.append(result)
    return results


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
        # AVG is handled via noisy SUM / noisy COUNT in the mechanism layer.
        # We return the SUM sensitivity here; the mechanism handles decomposition.
        return SensitivityResult(
            func="AVG",
            column=agg.column,
            sensitivity=bound,
            notes=f"Decomposed as noisy_sum/noisy_count. Bound: {bound}",
        )

    raise SensitivityError(f"Unsupported aggregate: {agg.func}")
