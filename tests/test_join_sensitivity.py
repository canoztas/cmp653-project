"""Soundness + sanity tests for elastic / smooth sensitivity of an equi-join COUNT.

The headline test is SOUNDNESS: on small constructed instances we compute the
TRUE local sensitivity of the join COUNT by brute force (add/remove every
possible single tuple and take the max change) and check that the elastic
sensitivity ES(0) is >= it. We also check smooth-sensitivity monotonicity and
the (eps,delta) noise calibration, and that the optional analyzer path does not
disturb the default d_max clamp.
"""

import math
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.join_sensitivity import (
    JoinSensitivity,
    compute_join_sensitivity,
    elastic_sensitivity,
    es_at,
    laplace_scale_from_smooth,
    recommended_beta,
    smooth_sensitivity,
)
from dpdb.analyzer import (
    SensitivityError,
    analyze_join_sensitivity_elastic,
    analyze_sensitivity,
)
from dpdb.config import Config
from dpdb.parser import parse_query


# --------------------------------------------------------------------------- #
# Brute-force ground truth for local sensitivity of an equi-join COUNT.
# --------------------------------------------------------------------------- #

def _join_count(left_keys, right_keys):
    """True COUNT(*) of left JOIN right ON left.key = right.key (inner equi-join)."""
    rc = Counter(right_keys)
    return sum(rc[k] for k in left_keys)


def _realized_local_sensitivity(left_keys, right_keys, universe):
    """Exact local sensitivity: max |COUNT(D') - COUNT(D)| over D' differing by one
    tuple (add any value from `universe` to either side, or remove any existing
    tuple from either side)."""
    base = _join_count(left_keys, right_keys)
    best = 0
    # remove an existing left/right tuple
    for i in range(len(left_keys)):
        d = left_keys[:i] + left_keys[i + 1:]
        best = max(best, abs(_join_count(d, right_keys) - base))
    for i in range(len(right_keys)):
        d = right_keys[:i] + right_keys[i + 1:]
        best = max(best, abs(_join_count(left_keys, d) - base))
    # add one tuple to left/right with any value from the universe
    for v in universe:
        best = max(best, abs(_join_count(left_keys + [v], right_keys) - base))
        best = max(best, abs(_join_count(left_keys, right_keys + [v]) - base))
    return best


def _mf(keys):
    return max(Counter(keys).values()) if keys else 0


# --------------------------------------------------------------------------- #
# SOUNDNESS: ES(0) >= realized local sensitivity on constructed instances.
# --------------------------------------------------------------------------- #

INSTANCES = [
    # (left_keys, right_keys, universe)
    ([1, 1, 2, 3], [1, 2, 2, 3], [1, 2, 3, 99]),
    ([1, 1, 1], [1, 1], [1, 2]),
    ([1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 2, 5, 7]),
    ([5, 5, 5, 5], [5], [5, 6]),
    ([], [1, 2, 3], [1, 2, 3, 4]),
    ([1, 2, 2, 2, 3, 3], [2, 2, 3, 3, 3, 4], [1, 2, 3, 4, 5]),
]


@pytest.mark.parametrize("left,right,universe", INSTANCES)
def test_elastic_es0_is_sound_upper_bound_on_local_sensitivity(left, right, universe):
    mf_l, mf_r = _mf(left), _mf(right)
    es0 = elastic_sensitivity(mf_l, mf_r)
    realized = _realized_local_sensitivity(left, right, universe)
    assert es0 >= realized, (
        f"ES(0)={es0} must upper-bound realized LS={realized} "
        f"(mf_l={mf_l}, mf_r={mf_r})")


@pytest.mark.parametrize("left,right,universe", INSTANCES)
def test_elastic_es0_equals_max_mf(left, right, universe):
    # ES(0) == LS == max(mf_left, mf_right) by construction.
    assert elastic_sensitivity(_mf(left), _mf(right)) == max(_mf(left), _mf(right))


@pytest.mark.parametrize("left,right,universe", INSTANCES)
def test_es_at_k_bounds_local_sensitivity_of_distance_k_neighbours(left, right, universe):
    """ES(k) must upper-bound the local sensitivity of EVERY database within
    Hamming distance k. Check k=1: add/remove one tuple, then measure LS there."""
    k = 1
    es_k = es_at(_mf(left), _mf(right), k)
    # enumerate distance-1 neighbours and check their realized LS <= ES(1)
    neighbours = []
    for i in range(len(left)):
        neighbours.append((left[:i] + left[i + 1:], right))
    for i in range(len(right)):
        neighbours.append((left, right[:i] + right[i + 1:]))
    for v in universe:
        neighbours.append((left + [v], right))
        neighbours.append((left, right + [v]))
    for nl, nr in neighbours:
        ls = _realized_local_sensitivity(nl, nr, universe)
        assert es_k >= ls, f"ES(1)={es_k} < LS={ls} at distance-1 neighbour"


# --------------------------------------------------------------------------- #
# Elastic sensitivity recovers sane values.
# --------------------------------------------------------------------------- #

def test_elastic_sane_values():
    assert elastic_sensitivity(7, 1) == 7.0     # TPC-H-like: <=7 lineitems/order, key unique in orders
    assert elastic_sensitivity(1, 1) == 1.0     # both keys unique -> single-row sensitivity
    assert elastic_sensitivity(0, 0) == 0.0     # empty
    assert es_at(3, 5, 0) == 5.0
    assert es_at(3, 5, 2) == 7.0                # +k


def test_elastic_rejects_negative():
    with pytest.raises(ValueError):
        elastic_sensitivity(-1, 2)
    with pytest.raises(ValueError):
        es_at(2, 2, -1)


# --------------------------------------------------------------------------- #
# Smooth sensitivity properties.
# --------------------------------------------------------------------------- #

def test_smooth_ge_local_and_brute_force_max():
    mf_l, mf_r, beta = 7, 1, 0.1
    ls0 = elastic_sensitivity(mf_l, mf_r)
    s, kstar = smooth_sensitivity(mf_l, mf_r, beta)
    # smooth sensitivity is an upper bound on local sensitivity (k=0 term)
    assert s >= ls0
    # matches a brute-force scan of g(k) = e^{-k beta}(ls0 + k)
    brute = max(math.exp(-k * beta) * (ls0 + k) for k in range(0, 2000))
    assert s == pytest.approx(brute, rel=1e-9)
    assert kstar >= 0


def test_smooth_decreases_with_beta():
    # larger beta -> stronger decay -> smaller (or equal) smooth sensitivity
    s_small, _ = smooth_sensitivity(7, 1, 0.05)
    s_large, _ = smooth_sensitivity(7, 1, 0.5)
    assert s_small >= s_large >= elastic_sensitivity(7, 1)


def test_smooth_requires_positive_beta():
    with pytest.raises(ValueError):
        smooth_sensitivity(7, 1, 0.0)
    with pytest.raises(ValueError):
        smooth_sensitivity(7, 1, -0.1)


def test_recommended_beta_and_laplace_scale():
    eps, delta = 1.0, 1e-6
    beta = recommended_beta(eps, delta)
    assert beta == pytest.approx(eps / (2 * math.log(2 / delta)))
    s, _ = smooth_sensitivity(7, 1, beta)
    scale = laplace_scale_from_smooth(s, eps, delta)
    assert scale == pytest.approx(2 * s / eps)
    assert scale > 0


def test_compute_join_sensitivity_bundle():
    js = compute_join_sensitivity(7, 1, epsilon=1.0, delta=1e-6,
                                  left_table="lineitem", right_table="orders")
    assert isinstance(js, JoinSensitivity)
    assert js.local_sensitivity == 7.0
    assert js.elastic_sensitivity == 7.0  # ES(0)
    assert js.smooth_sensitivity >= js.local_sensitivity
    assert js.beta == pytest.approx(recommended_beta(1.0, 1e-6))
    assert js.es_at(3) == 10.0


# --------------------------------------------------------------------------- #
# Analyzer wiring: optional path works, default d_max clamp is untouched.
# --------------------------------------------------------------------------- #

JOIN_SQL = ("SELECT COUNT(*) FROM orders JOIN lineitem ON l_orderkey = o_orderkey")


def test_default_dmax_clamp_unchanged():
    cfg = Config(fk_multiplicity={"orders": {"lineitem": 7}})
    parsed = parse_query(JOIN_SQL)
    res = analyze_sensitivity(parsed, cfg)
    assert len(res) == 1
    assert res[0].sensitivity == 7.0  # the conservative public clamp, unchanged


def test_optional_elastic_path_smooth_sensitivity():
    parsed = parse_query(JOIN_SQL)
    # mf(o_orderkey in orders)=1 (PK), mf(l_orderkey in lineitem)<=7
    res = analyze_join_sensitivity_elastic(parsed, mf_left=1, mf_right=7,
                                           epsilon=1.0, delta=1e-6)
    assert len(res) == 1
    assert res[0].func == "COUNT"
    # smooth sensitivity is reported; >= local sensitivity 7
    assert res[0].sensitivity >= 7.0
    assert "FLEX" in res[0].notes


def test_optional_elastic_path_rejects_non_join():
    parsed = parse_query("SELECT COUNT(*) FROM lineitem")
    with pytest.raises(SensitivityError):
        analyze_join_sensitivity_elastic(parsed, 1, 1, 1.0, 1e-6)
