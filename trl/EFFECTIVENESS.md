# Effectiveness benchmark

Generated 2026-11-27 12:28:11 from 4 lane files: `reports/trux_rep12/lane.jsonl` (trux-tsubame), `reports/trux_rep12/lane_timesfm.jsonl` (trux-tsubame), `reports/pass4_gpu7_full/lane_full66.jsonl` (gpu7), `reports/pass4_watson_arena/lane_arena.jsonl` (sedan-gpu1) — 2572 evaluation rows over 101 series, horizons 21, 63. Protocol: `docs/09_benchmark.md`.

Rows are **zero-shot** (no adaptation) and slide windows over the span(s) named above — for zero-shot rows the whole series is legitimate because the model saw no training data; an *adapted* checkpoint must be scored with `--spans test`, or the comparison leaks. Same horizon grid, same window count, same seed for every model. Rows carry the **registry key the sweep requested** (adapters may report a family-level key internally). `ctx` is the context actually used (an oversized one is clamped to the span, which is recorded, not hidden). **CRPS/WQL/pinball appear only for models that emit quantiles**; a point-only model gets point metrics and is marked `pt` — a probabilistic column with a dash is not a zero. `skill` is `1 − MASE` (MASE normalises by the in-sample mean |Δ| of the train span), so MASE is the column to read and a negative skill means the model lost to that trivial scaler on this series. `cov@80` is empirical coverage of the nominal 80% band (`nominal_80` in the JSON rows carries the nominal value the model actually claims, which is not always exactly 0.8 for models with their own quantile grid).

## Which interpreter produced which row

Every model runs in the venv its registry env names; a device column is there because CPU-versus-GPU changes the numbers. `?` means the field was not recorded by that run, or the interpreter path is relative to a different machine and cannot be probed from here ('not probed') — treat those rows as provenance-incomplete.

| interpreter | machine | device | torch (cuda?) | models |
|---|---|---|---|---|
| `/home/lujun.li/envs/tsfm-trl/core/bin/python` | sedan-gpu1 | cuda | 2.7.1+cu128 (cuda) | `arima`, `chronos-2`, `chronos-bolt-base`, `chronos-bolt-small`, `chronos-t5-base`, `chronos-t5-small`, `flowstate-r1`, `moving_average`, `naive`, `patchtst-fm-r2`, `seasonal_naive`, `theta`, `timesfm-2.5-200m`, `timesfm-3.0`, `ttm-r2`, `ttm-r3` |
| `/home/yiqun/envs/tsfm-trl/core/bin/python` | gpu7 | cuda | 2.9.1+cu130 (cuda) | `arima`, `chronos-2`, `chronos-bolt-base`, `chronos-bolt-small`, `chronos-t5-small`, `moving_average`, `naive`, `seasonal_naive`, `theta`, `timesfm-2.5-200m`, `ttm-r2` |

## Headline ranking

`beat naive` is the share of cells where the model's MASE beats the **naive row scored on the same (series, span, horizon) cell** - not MASE < 1: MASE's denominator is the in-sample mean |delta| of the train span, so on these series naive itself sits near 4.5 and a MASE < 1 test calls everything a loss. The ratio column is the median of per-cell `mase_model / mase_naive`, which is comparable across data paths (same bytes, same split, same denominator on both sides); raw median MASE is scoped to this pass. `ctx` prints the context lengths actually used - different ctx is a different measurement. `win` is the number of windows sampled per cell, and `--max-windows` draws them **uniformly at random** from the test span, so the window count is part of the measurement, not a knob: switching seed at 8 windows moves naive's own MASE ~55% median (max 100%) and the paired model/naive ratio ~12% (max 25%); at 32 windows those fall to ~21% and ~7% (measured 2026-11-27 on four arena series, same bytes, same code). A single cell within ~12% of parity is neither a win nor a loss - read aggregate shares, not cells. `no measurement` counts cells whose windows were all non-finite or unscored; those never enter a median.

| model_key | cells | beat naive | med MASE/naive | median MASE | median RMSE | ctx | win | no measurement |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| `arima` | 202 | 55% (112/202) | 0.999 | 5.08 | 1.62 | 256 | 8 | 0 |
| `chronos-bolt-small` | 202 | 41% (83/202) | 1.03 | 5.08 | 1.6 | 256 | 8 | 0 |
| `chronos-2` | 202 | 40% (80/202) | 1.02 | 5.22 | 1.59 | 256 | 8 | 0 |
| `chronos-t5-base` | 70 | 33% (23/70) | 1.05 | 5.27 | 1.37 | 256 | 8 | 0 |
| `chronos-bolt-base` | 202 | 32% (65/202) | 1.04 | 5.25 | 1.74 | 256 | 8 | 0 |
| `theta` | 202 | 30% (61/202) | 1.04 | 5.31 | 1.63 | 256 | 8 | 0 |
| `chronos-t5-small` | 202 | 30% (60/202) | 1.05 | 5.44 | 1.71 | 256 | 8 | 0 |
| `timesfm-2.5-200m` | 202 | 29% (59/202) | 1.06 | 5.32 | 1.74 | 256 | 8 | 0 |
| `timesfm-3.0` | 70 | 27% (19/70) | 1.08 | 5.49 | 1.34 | 256 | 8 | 0 |
| `ttm-r2` | 202 | 27% (54/202) | 1.25 | 6.15 | 2.11 | 512 | 8 | 0 |
| `seasonal_naive` | 202 | 19% (39/202) | 1.08 | 5.4 | 1.7 | 256 | 8 | 0 |
| `flowstate-r1` | 70 | 17% (12/70) | 1.31 | 6.45 | 1.5 | 1536 | 8 | 0 |
| `patchtst-fm-r2` | 70 | 16% (11/70) | 1.35 | 6.3 | 1.86 | 512 | 8 | 0 |
| `moving_average` | 202 | 13% (26/202) | 1.21 | 5.94 | 1.99 | 256 | 8 | 0 |
| `naive` | 202 | 0% (0/202) | 1 | 5.08 | 1.62 | 256 | 8 | 0 |
| `ttm-r3` | 70 | 0% (0/44) | 3.94 | 17.6 | 3.69 | 512 | 8 | 26 |

_Per-cell tables are omitted (`--summary-only`): a 3,600-row page is ~2 MB of HTML, and GitHub Pages does not serve the directory URL for a page that large - the full tables live in the repo's merged report instead._

## Repeated cell keys: 216

These `(model, series, span, horizon)` keys were measured more than once in the merge inputs; the table keeps the newest. Classifying every measurement pair by protocol signature (ctx, ctx_note, device, code_sha, torch): **216** pairs sit under a *different* signature - an earlier measurement superseded by a fix, where the gap is the size of that fix - and **0** share a signature, so their gap is run variance.

Superseded by a protocol change: naive (24), seasonal_naive (24), arima (24), chronos-2 (24), chronos-bolt-small (24), chronos-bolt-base (24)

Gap between a superseded cell and the one that replaced it: median 0.00%, max 17.0%. Some of these only differ by a code sha (gap ~0.0002%) and are plain re-runs; the large ones are a protocol fix, which is why the number is quoted and not averaged away.

Quote a model's ranking only where its gap to the next model exceeds the run-variance spread above.

## Not in this table (30 of 46 registry models)

No row was produced for these yet — they are **not** measured here, and their absence is not a pass:

`aurora`, `hirondel-7m`, `kairos-10m`, `kairos-50m`, `kronos-base`, `kronos-mini`, `kronos-small`, `moirai-1.1-R-base`, `moirai-1.1-R-small`, `moirai-2.0-R-small`, `moirai-moe-1.0-R-base`, `moment-1-large`, `moment-1-small`, `sundial-base-128m`, `t0-alpha`, `t0-beta`, `t0-gamma`, `timemoe-50m`, `timesfm-2.0-500m`, `timestral-123m`, `timestral-209m`, `timestral-28m`, `timestral-5m`, `tirex`, `tirex-2`, `toto-2-22m`, `toto-2-313m`, `toto-open-1.0`, `visiontspp-base`, `yug`

