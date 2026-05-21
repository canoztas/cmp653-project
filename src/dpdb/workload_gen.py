"""Zipf-parameterized workload generator over real TPC-H / Adult templates.

This module instantiates the analytical model (model.py) over actual SQL queries.
Each "template" corresponds to a concrete SQL string with a parametric slot;
sampling from a Zipf(alpha) distribution over templates produces a realistic
analytical workload whose privacy-budget consumption can be compared to the
analytical prediction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from dpdb.model import zipf_distribution


@dataclass
class WorkloadTemplate:
    """A SQL template with a callable that produces concrete query strings."""
    name: str
    description: str
    instantiate: Callable[[], str]  # produces concrete SQL each call


# --- Concrete template factories ---

def _tpch_lineitem_count_by_returnflag(flag: str) -> str:
    return f"SELECT COUNT(*) FROM lineitem WHERE l_returnflag = '{flag}'"


def _tpch_lineitem_sum_by_returnflag(flag: str) -> str:
    return f"SELECT SUM(l_extendedprice) FROM lineitem WHERE l_returnflag = '{flag}'"


def _orders_count_by_priority(prio: str) -> str:
    return f"SELECT COUNT(*) FROM orders WHERE o_orderpriority = '{prio}'"


def _adult_count_by_age_band(band_low: int) -> str:
    return f"SELECT COUNT(*) FROM adult WHERE age >= {band_low} AND age < {band_low + 10}"


def _adult_count_by_education(edu: str) -> str:
    return f"SELECT COUNT(*) FROM adult WHERE education = '{edu}'"


# --- Template registries ---

TPCH_RETURNFLAG_TEMPLATES = [
    WorkloadTemplate(
        name=f"lineitem_count_R{i}",
        description="COUNT lineitem by return flag",
        instantiate=(lambda f=flag: _tpch_lineitem_count_by_returnflag(f)),
    )
    for i, flag in enumerate(["R", "A", "N"])
]

ORDERS_PRIORITY_TEMPLATES = [
    WorkloadTemplate(
        name=f"orders_count_{p}",
        description=f"COUNT orders by priority {p}",
        instantiate=(lambda pp=p: _orders_count_by_priority(pp)),
    )
    for p in ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
]

ADULT_AGE_BAND_TEMPLATES = [
    WorkloadTemplate(
        name=f"adult_age_{low}_{low+10}",
        description=f"COUNT adult age in [{low}, {low+10})",
        instantiate=(lambda l=low: _adult_count_by_age_band(l)),
    )
    for low in [20, 30, 40, 50, 60, 70, 80]
]

ADULT_EDUCATION_TEMPLATES = [
    WorkloadTemplate(
        name=f"adult_edu_{edu}",
        description=f"COUNT adult by education = {edu}",
        instantiate=(lambda e=edu: _adult_count_by_education(e)),
    )
    for edu in ["Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college"]
]


# --- Workload generation ---

def generate_zipf_workload(
    templates: list[WorkloadTemplate],
    alpha: float,
    k: int,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Generate a sequence of k queries by sampling templates from Zipf(alpha).

    Returns:
        (queries, template_indices) — concrete SQL strings and the indices of
        the templates they came from.
    """
    rng = np.random.default_rng(seed)
    p = zipf_distribution(len(templates), alpha)
    indices = rng.choice(len(templates), size=k, p=p)
    queries = [templates[i].instantiate() for i in indices]
    return queries, list(indices)


def generate_repetitive_workload(template: WorkloadTemplate, k: int) -> list[str]:
    """W1-style: same query repeated k times. Corresponds to alpha -> inf limit."""
    return [template.instantiate() for _ in range(k)]


def generate_uniform_workload(
    templates: list[WorkloadTemplate], k: int, seed: int = 42
) -> tuple[list[str], list[int]]:
    """W3-style: uniform sampling. Corresponds to alpha = 0 limit."""
    return generate_zipf_workload(templates, alpha=0.0, k=k, seed=seed)


# --- Built-in workload families (named, reproducible) ---

WORKLOAD_FAMILIES = {
    "W1_repetitive":   ("Adult age 30-40 only", lambda k, seed: (
        generate_repetitive_workload(ADULT_AGE_BAND_TEMPLATES[1], k),
        [1] * k,
    )),
    "W2_zipf_a1.0":   ("Adult age bands, Zipf(α=1.0)", lambda k, seed: (
        generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES, alpha=1.0, k=k, seed=seed)
    )),
    "W2_zipf_a0.5":   ("Adult age bands, Zipf(α=0.5)", lambda k, seed: (
        generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES, alpha=0.5, k=k, seed=seed)
    )),
    "W2_zipf_a2.0":   ("Adult age bands, Zipf(α=2.0)", lambda k, seed: (
        generate_zipf_workload(ADULT_AGE_BAND_TEMPLATES, alpha=2.0, k=k, seed=seed)
    )),
    "W3_uniform":     ("Adult age bands, uniform", lambda k, seed: (
        generate_uniform_workload(ADULT_AGE_BAND_TEMPLATES, k, seed=seed)
    )),
    "W4_drilldown":   ("Progressive narrowing (no repeats)", lambda k, seed: _drilldown(k, seed)),
}


def _drilldown(k: int, seed: int) -> tuple[list[str], list[int]]:
    """Drill-down workload: each query strictly narrower than the previous.

    Models the differencing-attack pattern; no exact repeats by design.
    """
    rng = np.random.default_rng(seed)
    queries = []
    indices = []
    for i in range(k):
        # Each drill-down has a unique combination of WHERE clauses
        age_low = 20 + (i % 7) * 10
        flag = ["R", "A", "N"][i % 3]
        q = f"SELECT SUM(l_extendedprice) FROM lineitem WHERE l_returnflag = '{flag}'"
        queries.append(q)
        indices.append(i)  # every query has its own "template" index for drill-down
    return queries, indices


def get_workload(name: str, k: int, seed: int = 42) -> tuple[list[str], list[int]]:
    """Convenience: produce queries + template indices by family name."""
    if name not in WORKLOAD_FAMILIES:
        raise KeyError(f"Unknown workload: {name}")
    desc, factory = WORKLOAD_FAMILIES[name]
    return factory(k, seed)
