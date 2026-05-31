"""Tests for the alternative u_k predictors (unseen-species estimators)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.predictors import (
    frequency_of_frequencies,
    predict_chao1,
    predict_good_toulmin,
    predict_plugin,
    predict_smoothed_gt,
)


class TestFreqOfFreqs:
    def test_counts(self):
        # three templates seen 1,1,3 times -> phi = {1:2, 3:1}
        phi = frequency_of_frequencies([1, 1, 3])
        assert phi == {1: 2, 3: 1}

    def test_ignores_zero(self):
        assert frequency_of_frequencies([0, 2, 0]) == {2: 1}


class TestPredictors:
    def test_plugin_bounded_by_seen(self):
        # plug-in support is only over observed templates -> never exceeds count
        counts = [5, 3, 2]
        assert predict_plugin(counts, n=10, k=100) <= len(counts) + 1e-6

    def test_good_toulmin_zero_horizon(self):
        # t=0 (k==n) -> predicts exactly the distinct count seen
        counts = [4, 3, 3]
        assert abs(predict_good_toulmin(counts, n=10, k=10) - 3) < 1e-9

    def test_good_toulmin_extrapolates_up(self):
        # with singletons, extending the sample predicts MORE than seen
        counts = [1, 1, 1, 5]  # 3 singletons -> new species expected
        assert predict_good_toulmin(counts, n=8, k=12) > 4  # > distinct seen

    def test_smoothed_gt_finite_large_horizon(self):
        # raw GT diverges for t>1; smoothed variant must stay finite
        counts = [1, 1, 1, 1, 1, 2, 3]
        val = predict_smoothed_gt(counts, n=9, k=90)  # t ~ 9
        assert val == val and val < float("inf")  # not NaN/inf

    def test_chao1_at_least_seen(self):
        counts = [1, 1, 2, 5]
        assert predict_chao1(counts, n=9, k=50) >= 3  # >= distinct seen
