"""Quantile metrics must know which axis is which, and the seasonal-naive denominator must be seasonal.

Both of these were found in review of the cherry-picked benchmark tree, and both fail silently: a
square (9 levels x 9 horizons) matrix passes any orientation check and produces a plausible CRPS
computed across the wrong axis, and a seasonal-naive denominator taken as a block of older history
makes a "relative to naive" number move when data the model never saw changes.
"""
from __future__ import annotations

import numpy as np

from tsfm_eval.metrics import (QUANTILE_LEVELS, crps_from_quantiles, normalized_difference,
                               quantile_coverage)

K = len(QUANTILE_LEVELS)


def _hk(levels_first: bool = False) -> np.ndarray:
    """A monotone quantile fan: level k rises with k, horizon t drifts with t."""
    q = np.array([[1.0 * t + 0.5 * k for k in range(K)] for t in range(K)])   # (H=9, K=9)
    return q.T if levels_first else q


def test_a_square_quantile_matrix_is_not_guessed():
    """H == K == 9 with the default grid: `len(levels)` matching cannot tell the two layouts apart,
    so the old auto-detect transposed a correct (H, K) input and scored pinball across horizons."""
    q = _hk()
    assert q.shape == (K, K)
    truth = np.arange(1.0, K + 1.0)
    hk = crps_from_quantiles(truth, q, layout="HK")
    kh = crps_from_quantiles(truth, q.T, layout="KH")
    assert np.isfinite(hk) and np.isfinite(kh)
    wrong = crps_from_quantiles(truth, q, layout="KH")
    assert hk != wrong, "the two layouts must not agree, or the test cannot see the bug"
    assert np.isnan(crps_from_quantiles(truth, q, layout="auto")), (
        "auto must refuse a square input instead of picking one")


def test_coverage_uses_the_declared_layout_and_a_real_level():
    q = _hk()
    truth = np.arange(1.0, K + 1.0)
    # the 10th percentile column of an (H, K) matrix is q[:, 0]
    want = float(np.mean(truth < q[:, 0]))
    assert np.isclose(quantile_coverage(truth, q, level=0.1), want)
    # a level the grid does not carry is a measurement that does not exist, not a nearest neighbour
    assert np.isnan(quantile_coverage(truth, q, level=0.05))


def test_the_seasonal_naive_denominator_repeats_the_last_season():
    """history [10, 11, 12], season 1, truth [13, 14]: seasonal naive predicts [12, 12]. A block of
    older history ([11, 12]) is not seasonal naive - it is a different lag, and it changes when
    irrelevant history changes."""
    nd = normalized_difference(np.array([13.0, 14.0]), np.array([13.0, 14.0]),
                               history=np.array([10.0, 11.0, 12.0]), season=1)
    assert np.isclose(nd, 0.0), nd            # a perfect forecast scores 0 against [12, 12]
    # the denominator must not depend on how much older history is supplied
    nd2 = normalized_difference(np.array([13.0, 14.0]), np.array([13.0, 14.0]),
                               history=np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]), season=1)
    assert np.isclose(nd2, 0.0)
    # season 2: the last two observations repeat, in order
    v = normalized_difference(np.array([99.0, 99.0]), np.array([10.0, 20.0]),
                              history=np.array([5.0, 7.0, 9.0]), season=2)
    # yhat[T+1] = y[T+1-season]: with history ending at 9, that is [7, 9].
    # MAE model = mean(|99-10|, |99-20|) = 84; MAE naive = mean(|7-10|, |9-20|) = 7
    assert np.isclose(v, 84.0 / 7.0), v
