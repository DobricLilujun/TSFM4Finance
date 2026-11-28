"""Cost-aware trading metrics, checked against hand-computed numbers.

These landed in the canonical metrics package rather than in the arena tree because both
`tsfm4finance.bench` and the unified evaluator import them, and two copies of a Sharpe ratio is
how a table ends up reporting two of them. The numbers below are worked out on paper, so a
scaling mistake (bps is 1e-4, not 1e-3) fails here instead of in a published row.
"""
from __future__ import annotations

import numpy as np

from tsfm_eval.metrics import TRADING_COST_GRID, ic_signal, max_drawdown, trading_metrics


def test_ten_basis_points_is_one_thousandth_of_turnover():
    # hold +1 for four periods on +1% log returns each: one unit of entry turnover, nothing after
    m = trading_metrics(np.ones(4), np.full(4, 0.01), cost_bps=0.0, periods_per_year=252)
    assert m["ann_return_gross"] == round(0.01 * 252, 10)
    assert m["ann_return_net"] == m["ann_return_gross"]
    m10 = trading_metrics(np.ones(4), np.full(4, 0.01), cost_bps=10.0, periods_per_year=252)
    # cost = 1.0 turnover * 10/10_000 = 0.001, paid in one period out of four
    assert np.isclose(m10["ann_return_net"], (0.04 - 0.001) / 4 * 252)
    assert m10["turnover_total"] == 0.25


def test_a_position_that_ignores_the_sign_of_the_return_scores_zero_hit_rate():
    m = trading_metrics(np.ones(3), np.array([0.01, -0.02, 0.03]), cost_bps=0.0)
    assert np.isclose(m["hit_rate"], 2 / 3)
    assert m["n_bets"] == 3.0


def test_drawdown_is_a_positive_fraction_of_a_simple_return_stream():
    # -50% then +100% then flat: wealth goes 1 -> 0.5 -> 1 -> 1, so the trough is half.
    # The unanchored version of this (peak over cumprod only, no initial 1.0) reads that as 0.0:
    # the first period's loss is compared against itself. A window that opens with a drawdown
    # - which is exactly what a risk metric is for - reported no risk at all.
    assert np.isclose(max_drawdown(np.array([-0.5, 1.0, 0.0])), 0.5)
    assert np.isclose(max_drawdown(np.array([-0.5, 0.1])), 0.5)
    assert max_drawdown(np.zeros(4)) == 0.0
    # a gain first then a fall: anchored and unanchored agree here, which is why the bug hid
    assert np.isclose(max_drawdown(np.array([0.5, -0.75])), 0.75)
    assert np.isnan(max_drawdown(np.array([])))


def test_ic_is_a_correlation_and_is_nan_when_it_cannot_be_one():
    assert np.isclose(ic_signal(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])), 1.0)
    assert np.isnan(ic_signal(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


def test_the_cost_grid_the_reports_iterate_is_public():
    assert TRADING_COST_GRID == [0.0, 10.0, 30.0]


def test_the_arena_shim_and_the_package_agree():
    """`tsfm4finance.core.metrics` is a re-export shim; a divergence there is invisible until a
    benchmark row disagrees with itself."""
    import tsfm4finance.core.metrics as shim
    for name in ("trading_metrics", "max_drawdown", "ic_signal", "TRADING_COST_GRID"):
        assert getattr(shim, name) is getattr(__import__("tsfm_eval.metrics", fromlist=[name]), name)
