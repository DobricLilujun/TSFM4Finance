# 7 · Financial tasks, panels and leakage

The label data here is finance-derived (plus a tiny public UEA/UCR subset for external
validity), so the rules that keep the numbers meaningful are leakage rules.

## Dataset layout

A dataset directory is the most useful form because it carries the split you intend:

```
data/sp500_daily/
  meta.json          # {"name": "...", "target": "close", "freq": "1D"}
  train.parquet      # ┐ concatenated in this order; the split boundaries are
  validation.parquet   ├ preserved as index ranges, so an adapted model can be
  test.parquet       # ┘ proven never to have seen the evaluation span
```

A single `.csv`/`.parquet` works too (70/15/15 split), as does a DataFrame, a
`{train,validation,test}.npz`, or a raw 1-D array.

The target column is `meta.json["target"]`, else the first of
`close/target/y/value/price`, else the last column.

## Panels and the price-column rule

`SeriesDataset` keeps the target as `values` and everything else as `panel`. Two rules
that took a failed model run to learn:

* `panel_window(start, stop)` **re-attaches the target column**. The panel excludes the
  target by construction, but a panel-consuming model (Kronos) asks for
  `open/high/low/close` and fails with `Price columns ['open','high','low','close'] not
  found` if `close` is missing from what it is handed.
* A registry row declares what it needs in `needs` (e.g. `("close",)`); rows with the
  `panel_in` capability get the DataFrame instead of a 1-D slice. A model that needs
  OHLCV cannot be fed a bare close series, and says so.

For fine-tuning, a panel-only vendor loop (Kronos' `finetune_csv`) synthesises the flat
columns from the target and records that it did — a synthetic panel is disclosed in the
checkpoint, not hidden.

## Label tasks

`dir`/`dir5` (direction with a dead band), `extreme` (move beyond *k* conditional
sigmas), `volreg` (forward volatility regime) are defined in
[`04_classify_probing.md`](04_classify_probing.md). All of them use **rolling,
backward-looking** scale estimates: the band in `dir5` is
`band × rolling-20-step log-return sd` at the *decision* point, never the in-sample
global sd. That is what lets one rule be applied to S&P and BTC without retuning.

## Splits and CV

* `ClassificationDataset.split` is **time-ordered** (0 train, 1 test). Shuffled windows
  over a price series leak the future into the training set.
* Probes are evaluated with `folds(split, mode="expanding")`: each fold trains on the
  past and predicts the next block, like a backtest.
* `train_windows(..., split="train")` records `split_end_index` (and the timestamp in
  `split_end`), and every checkpoint stores it. `tsfm-trl ckpt <path>` prints it, so an
  adapted model's evaluation can be audited for the leak before anyone believes its
  skill numbers.

## Anchor-relative metrics

On price panels, absolute MAE is dominated by the series' level and by volatility
regimes, so the default comparisons are relative:

* `nd` — normalised error against a naive anchor (last value);
* `mase` — scaled by the in-sample naive error;
* `relative_skill(model_row, baseline_row)` — versus a *registered baseline*
  (seasonal naive, theta, ARIMA), which is why baselines are first-class registry rows;
* directional accuracy and `rank_corr` are reported per row, but only mean something
  next to the class base rate (a 60 % up-majority task makes 55 % accuracy *worse*).

The reward side of the library encodes the same idea structurally: the reward callable
receives `anchor` (the last observed price) precisely so a P&L-style reward cannot be
written to accidentally reward "forecast the context mean".

## What this library is not

It is not a backtester: no costs, no position sizing, no portfolio accounting. The
finance labels exist to give the models a hard, leakage-prone task at small scale. If
you want tradable claims, bring your own walk-forward harness — and use the
`split_end` provenance so it cannot overlap the training span.
