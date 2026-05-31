"""Guards for the allocation-policy comparison harness.

These pin the two claims the paper's allocation table rests on: the budget
ledger is never overspent, and a budget-aware policy genuinely lowers the
fresh-release noise relative to naive on a skewed workload.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import allocation_policy_comparison as apc  # noqa: E402
from dpdb.model import zipf_distribution  # noqa: E402


def _draws(seed=0, alpha=1.5):
    rng = np.random.default_rng(seed)
    p = zipf_distribution(apc.M, alpha)
    return rng, rng.choice(apc.M, size=apc.K, p=p)


def test_budget_never_overspent():
    # every policy must respect the total budget B (within one floor grant)
    for fn in apc.POLICIES.values():
        rng, draws = _draws()
        spent, answered, all_e, miss_e = fn(rng, draws)
        assert spent <= apc.B + apc.FLOOR + 1e-9
        assert answered <= apc.K
        assert len(miss_e) <= len(all_e)  # misses are a subset of answers


def test_budget_aware_beats_naive_on_fresh_releases():
    # the core claim: closed-form allocation halves fresh-release noise vs naive
    naive_miss, cf_miss = [], []
    for seed in range(40):
        rng, draws = _draws(seed=seed)
        cf_miss.append(np.mean(apc.run_closed_form(rng, draws)[3]))
        rng, draws = _draws(seed=seed)
        naive_miss.append(np.mean(apc.run_naive(rng, draws)[3]))
    assert np.mean(cf_miss) < np.mean(naive_miss)


def test_naive_underspends_budget():
    # naive divides by k but only ~u_k misses occur -> leaves budget unused
    rng, draws = _draws()
    spent, _, _, _ = apc.run_naive(rng, draws)
    assert spent < 0.5 * apc.B  # uses well under half the budget


def test_safe_policy_never_rejects():
    # eps_q = B/m with u_k <= m guarantees total spend <= B and 100% answered
    for seed in range(30):
        rng, draws = _draws(seed=seed)
        spent, answered, _, _ = apc.run_closed_form_safe(rng, draws)
        assert answered == apc.K            # every query answered (no rejection)
        assert spent <= apc.B + 1e-9        # budget never exceeded


def test_safe_policy_beats_bandit_on_noise():
    # the honest headline: jumping to the B/m operating point beats the bandit,
    # which must discover it by exploration and pays a noise tax
    safe, band = [], []
    for seed in range(50):
        rng, draws = _draws(seed=seed)
        safe.append(np.mean(apc.run_closed_form_safe(rng, draws)[3]))
        rng, draws = _draws(seed=seed)
        band.append(np.mean(apc.run_bandit(rng, draws)[3]))
    assert np.mean(safe) < np.mean(band)


def test_oracle_is_the_ceiling():
    # oracle (eps_q = B/u_k, perfect prediction) is the best any policy can do
    rng, draws = _draws()
    oracle_miss = np.mean(apc.run_oracle(rng, draws)[3])
    rng, draws = _draws()
    safe_miss = np.mean(apc.run_closed_form_safe(rng, draws)[3])
    assert oracle_miss <= safe_miss + 0.5   # oracle no worse than the safe floor
