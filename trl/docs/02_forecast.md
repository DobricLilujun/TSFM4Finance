# 2 · Forecasting and evaluation

## One call, one contract

```python
from tsfm_trl import ModelHub
m = ModelHub.load("chronos-2", device="cuda", ctx_cap=2048, batch=16)
pred = m.predict([ctx_a, ctx_b], horizon=63)
```

`Pred` is the same object for every family:

| field | shape | |
|---|---|---|
| `point` | `(B, H)` | median/mean forecast, always |
| `quantiles` | `(B, H, len(LEVELS))` or `None` | resampled onto the **canonical grid** `LEVELS` |
| `samples` | `(B, K, H)` or `None` | sample paths when the model exposes them |
| `meta` | dict | how the interval was produced, normalisation, backend/kernel choice |

```python
from tsfm_trl.model.base import LEVELS     # (0.1 … 0.9) canonical grid
pred.quantile(0.9)                          # (B, H) column of the canonical grid
```

Adapters call `to_levels(q, their_levels)` on the way out, so TiRex's 100-quantile grid,
TimesFM's 9-quantile grid and Moirai's 0.1-0.9 head all become comparable. A model that
emits no distribution sets `quantiles = None` (MOMENT's mean head, small TTM releases,
the statistical baselines) rather than inventing one.

Contexts are trimmed by `trim()` to `max_context`, aligned to the model's patch step,
and pandas objects are coerced with `data.series.as_1d` (a `DataFrame` under
`np.asarray` iterates *column names*, which is where "could not convert string to
float: 'close'" comes from).

## Evaluation

```python
from tsfm_trl.eval.evaluate import forecast_eval
from tsfm_trl.eval.metrics import forecast_report

rows = forecast_eval(m, ds, horizons=(21, 63), spans=("test",), max_windows=64)
print(forecast_report(rows))
```

Point metrics: `mae`, `rmse`, `mape`, `smape`, `mase`, `nd`, `directional_accuracy`,
`rank_corr`. Distribution metrics: `pinball`, `wql`, `crps_from_quantiles`,
`quantile_coverage`, and `relative_skill` to compare against a baseline row.

Two honesty rules:

* **`quantile_coverage` counts only the bands the model emits.** A model whose grid
  stops at 0.9 must not be credited with 95 % coverage.
* **Anchor-relative error (`nd`, `mase`) is the default comparison**, because on price
  panels absolute MAE is dominated by the series' own scale. See
  [`07_finance_tasks.md`](07_finance_tasks.md).

## Multi-step longer than one patch

Patch-output models (TiRex/TiRex-2) produce a fixed patch per forward. The adapters do
a **median-rollback rollout** — forecast, append the median, forecast again — and
record it:

```python
pred.meta["rollout_steps"]   # e.g. 3 for horizon 96 with a 32-step patch
```

so a table reader knows a 96-step TiRex row is a rollout, not a single decode.

## Baselines are part of the matrix

`naive`, `seasonal_naive`, `theta`, `arima`, `moving_average` are registered models in
the same registry. They exist so a "✅ fine-tune" on a 300 M TSFM can be compared with
something that costs nothing, and so a metric bug shows up immediately (a TSFM that
loses to seasonal naive on a synthetic fixture is a measurement bug). `theta` is
implemented in-house because statsmodels never shipped `ThetaModel` despite promising
it since 0.12.

## Anomaly

`m.anomaly_scores(contexts)` is model-native when the family has a reconstruction or
residual path (MOMENT, Moirai, Toto), and otherwise a documented fallback: the
forecast residual under the model's own interval (or a context-scaled z-score when the
model has no interval), with `supports["anomaly"]` stating which one produced it.
