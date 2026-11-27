#!/usr/bin/env python3
"""Zero-shot forecast + honest metrics.    python examples/01_forecast.py [model]

`forecast_eval` returns one dict per (span, horizon) with the metrics already
computed; a bare MAE on a price panel says nothing, so the interesting columns are
the anchor-relative ones (`nd`, `mase`, `relative_skill`) read *next to a registered
baseline row*.
"""
import sys

from tsfm_trl import ModelHub
from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.eval.evaluate import forecast_eval

model_key = sys.argv[1] if len(sys.argv) > 1 else "chronos-2"
ds = tiny_finance(3000, 600)
KEYS = ("rmse", "mae", "mase", "smape", "nd", "relative_skill", "pinball", "crps")


def run(key, device):
    m = ModelHub.load(key, device=device, ctx_cap=512, batch=8)
    rows = forecast_eval(m, ds, horizons=(21,), spans=("test",), ctx=256, max_windows=32)
    return m, rows[0]


m, row = run(model_key, "auto")
_, base = run("seasonal_naive", "cpu")

print(f"{'metric':16s} {model_key:>18s} {'seasonal_naive':>16s}")
for k in KEYS:
    a, b = row.get(k), base.get(k)
    fmt = lambda v: "     —" if v is None or v != v else f"{v:9.4f}"
    print(f"{k:16s} {fmt(a):>18s} {fmt(b):>16s}")

p = m.predict([ds.values[:256]], horizon=21)
print("\npoint", p.point.shape,
      "quantiles", None if p.quantiles is None else p.quantiles.shape)
print("how the interval was produced:", p.meta)
