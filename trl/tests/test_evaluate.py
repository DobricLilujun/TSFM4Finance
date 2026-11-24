"""Evaluation-side behaviour that a real model would hide.

The panel test is the important one: a covariate frame misaligned with its context
still *runs*, so a benchmark row for a panel-fed model would have been scored on the
wrong covariates. Everything here is pure numpy on the synthetic fixture.
"""
from __future__ import annotations

import numpy as np

from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.eval.evaluate import forecast_eval
from tsfm_trl.eval.metrics import forecast_report
from tsfm_trl.model.base import Pred

H = 8


class _PanelModel:
    """Feeds the panel through and records exactly what it was handed."""

    key = "stub_panel"
    caps = ("panel_in",)
    max_context = 512
    ctx_step = 1
    adaptation = "zero"

    def __init__(self):
        self.panels = []
        self.seen = []

    def predict(self, contexts, horizon, panels=None):
        ps = panels or []
        assert len(ps) == len(contexts), "panel count != context count"
        self.panels.extend(ps)
        self.seen.extend((np.asarray(c, dtype=float), p) for c, p in zip(contexts, ps))
        pts = np.asarray([np.asarray(c)[-horizon:] for c in contexts], dtype=float)
        return Pred(point=pts, meta={"how": "repeat the last h points"})

def _ds(n=600):
    return tiny_finance(n)


def test_forecast_eval_clamps_an_oversized_context_instead_of_raising():
    ds = _ds(600)
    m = _PanelModel()
    rows = forecast_eval(m, ds, horizons=(H,), ctx=4096, max_windows=4)
    assert len(rows) == 1
    r = rows[0]
    assert r["ctx"] < 4096 and r["ctx"] > 0
    assert "clamped" in r["ctx_note"]
    assert r["n_windows"] > 0


def test_panel_rows_are_aligned_with_their_context():
    """The covariate frame handed to a panel-fed model must be the same span as its
    context. A misaligned panel still runs, so this is the kind of benchmark bug that
    never shows up as a failure — it just silently scores the wrong covariates."""
    ds = _ds(600)
    m = _PanelModel()
    ctx = 48
    rows = forecast_eval(m, ds, horizons=(H,), ctx=ctx, max_windows=5)
    assert m.seen, "the model asked for panels and got none"
    assert len(m.seen) == rows[0]["n_windows"]
    offsets = []
    for ctx_vals, pan in m.seen:
        assert len(pan) == ctx, "panel length must match the context"
        col = np.asarray(pan[ds.target], dtype=float)
        assert np.allclose(col, ctx_vals), "panel span != context span"
        hits = [k for k in range(len(ds.values) - ctx + 1)
                if np.allclose(np.asarray(ds.values)[k:k + ctx], col)]
        assert hits, "the panel is not a slice of the series at any offset"
        offsets.append(hits[0])
    assert len(set(offsets)) == len(offsets), "two windows share one panel offset"


def test_forecast_report_carries_direction_and_rank_metrics():
    y = np.array([[10.0, 11.0, 12.0], [20.0, 19.0, 18.0]])
    good = Pred(point=y.copy(), meta={})
    rep = forecast_report(y, good, np.linspace(9, 21, 40), anchor=[10.0, 20.0])
    for k in ("rmse", "mae", "mase", "smape", "nd", "relative_skill", "dir_acc", "rank_corr"):
        assert k in rep, k
    assert rep["dir_acc"] == 1.0
    assert rep["rank_corr"] == 1.0


def test_direction_is_measured_against_the_anchor_when_given():
    y = np.array([[5.0, 6.0]])          # both truths above the anchor
    p = np.array([[4.0, 9.0]])          # first point below the anchor: one wrong sign
    rep = forecast_report(y, Pred(point=p, meta={}), np.arange(20.0),
                          anchor=[5.0])
    assert abs(rep["dir_acc"] - 0.5) < 1e-9
