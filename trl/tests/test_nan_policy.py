"""A model that emits NaN must not hand NaN to the metrics.

TTM r3 does this in practice: the same 512-point window came back clean in one process and
all-NaN in the next. The harness retries the affected windows one at a time and, if they
are still non-finite, reports the cell as a measurement failure with the metric columns set
to None - because a NaN in a metric column is read as a number by anything that aggregates
it, and the report's medians turn into NaN without saying why.
"""
from __future__ import annotations

import numpy as np

from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.eval.evaluate import _finite_or_none, forecast_eval
from tsfm_trl.model.base import Pred


class _Flaky:
    """NaN at window 1. With ``batch_only`` the window is fine when asked alone, which is
    what the single-window retry exists for; without it, the NaN survives the retry and the
    cell has to be reported as a failure."""

    key = "stub_flaky"
    caps = ("forecast",)
    max_context = 2048
    min_context = 0
    ctx_step = 1
    adaptation = "zero"

    def __init__(self, *, batch_only=False, poison=True):
        self.batch_only = batch_only
        self.poison = poison
        self.calls = []

    def predict(self, contexts, horizon, panels=None):
        n = len(contexts)
        self.calls.append(n)
        out = np.zeros((n, horizon), dtype=float)
        for i, c in enumerate(contexts):
            c = np.asarray(c, dtype=float)
            # poisoned at window 1 in the batch; when batch_only is False it is poisoned
            # again when the harness re-asks for that window alone
            if self.poison and ((n > 1 and i == 1) or (n == 1 and not self.batch_only)):
                out[i] = np.nan
                continue
            out[i] = c[-horizon:] if len(c) >= horizon else np.zeros(horizon)
        return Pred(point=out, meta={"how": "stub"})


def _rows(model, n_windows=4):
    ds = tiny_finance(1200)
    return forecast_eval(model, ds, horizons=(8,), spans=("test",), ctx=128,
                         max_windows=n_windows)


def test_a_clean_model_leaves_no_nan_fields():
    rows = _rows(_Flaky(poison=False))
    assert rows and rows[0]["mase"] is not None
    assert "pred_nan_windows" not in rows[0]
    assert rows[0]["seconds"] >= 0


def test_a_batch_only_nan_is_repaired_by_a_single_window_retry():
    m = _Flaky(batch_only=True)
    rows = _rows(m)
    r = rows[0]
    assert "pred_nan_windows" not in r, f"retry should have fixed it: {r.get('pred_nan_windows')}"
    assert r["mase"] is not None
    assert m.calls[0] == 4, "the first call is the whole batch"
    assert m.calls[1:] == [1], f"retry must be per window, not a batch rerun: {m.calls}"


def test_a_window_that_stays_nan_leaves_the_scoring_set():
    m = _Flaky(batch_only=False)
    r = _rows(m, n_windows=4)[0]
    assert r.get("pred_nan_windows") == 1, r.get("pred_nan_windows")
    assert "non-finite" in (r.get("error") or ""), r.get("error")
    # the honest part: the denominator is the scored count, not the requested one
    assert r["n_windows"] == 3, f"expected 3 scored windows, got {r['n_windows']}"
    assert r["mase"] is not None, "the surviving windows are still a measurement"


def test_non_finite_metrics_become_none():
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(-float("inf")) is None
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(None) is None
