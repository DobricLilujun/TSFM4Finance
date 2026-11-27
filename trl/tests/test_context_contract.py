"""The context-length contract: some checkpoints only accept the length they declare.

Measured on granite-timeseries-ttm-r3 (declares context_length=512): fed 256 it pads the
window with zeros itself and the forecast level comes back at 0.50x the real level (0.25x at
128, 0.99x at 512). Under the benchmark protocol that is a MASE of ~13x naive for a model
whose weights are fine -- the number measures our window, not the model. So the harness
raises the protocol context to the declared one, and the adapter pads with the oldest
observation rather than trusting vendor zero-padding when the span cannot supply it.
"""
from __future__ import annotations

import numpy as np

from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.eval.evaluate import forecast_eval
from tsfm_trl.model.adapters.granite import pad_left_to
from tsfm_trl.model.base import Pred, TSFMModel


def test_pad_left_to_repeats_the_oldest_value_and_reports_it():
    cs = [np.arange(4.0), np.arange(10.0)]
    out, added = pad_left_to(cs, 6)
    assert added == 2
    assert list(out[0]) == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0], "pad must be flat, not zero-filled"
    assert out[1] is not None and out[1].shape[0] == 10, "a long enough context is untouched"


def test_pad_left_to_is_a_noop_without_a_contract():
    cs = [np.arange(3.0)]
    out, added = pad_left_to(cs, 0)
    assert added == 0 and np.array_equal(out[0], cs[0])


class _ContractModel:
    """A model that refuses to be scored on a short window: it records the shortest context
    it was actually handed."""

    key = "stub_contract"
    caps = ("forecast",)
    max_context = 2048
    min_context = 512
    ctx_step = 1
    adaptation = "zero"

    def __init__(self):
        self.seen = []

    def predict(self, contexts, horizon, panels=None):
        self.seen.append(min(len(np.asarray(c)) for c in contexts))
        pts = np.asarray([[np.asarray(c, dtype=float)[-1]] * horizon for c in contexts])
        return Pred(point=pts, meta={"how": "last value"})


def test_forecast_eval_respects_a_declared_context():
    ds = tiny_finance(4000)
    m = _ContractModel()
    rows = forecast_eval(m, ds, horizons=(8,), spans=("test",), ctx=256, max_windows=2)
    assert rows, "the fixture span should be long enough for the raised context"
    r = rows[0]
    assert r["ctx"] == 512, f"expected the contract window, got {r['ctx']}"
    assert "raised" in (r.get("ctx_note") or ""), "the row must say the protocol ctx was raised"
    assert m.seen and m.seen[0] >= 512, "the contexts handed to predict must honour the contract"


def test_models_without_a_contract_are_untouched():
    class _Open(_ContractModel):
        min_context = 0

    ds = tiny_finance(4000)
    m = _Open()
    rows = forecast_eval(m, ds, horizons=(8,), spans=("test",), ctx=256, max_windows=2)
    assert rows[0]["ctx"] == 256
    assert not rows[0].get("ctx_note"), "no contract, no note, no change"


def test_the_base_class_declares_the_contract_field():
    assert TSFMModel.min_context == 0, "every adapter must be able to opt out explicitly"
