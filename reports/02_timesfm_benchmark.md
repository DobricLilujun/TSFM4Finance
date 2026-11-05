# Google TimesFM on real financial series — zero-shot, few-shot, fine-tuned, classification, anomaly and quant-trading evaluation

Protocol: **spec §4.1** — rolling-origin backtest, non-overlapping windows (stride = horizon), expanding context capped at 512, MASE scale from the train split (season = 1), horizons h ∈ {1, 5, 12, 21, 63} business days.

Model: **Google TimesFM 2.5-200M** (`timesfm` 2.0.2 pip, PyTorch checkpoint `google/timesfm-2.5-200m-pytorch`), full fine-tuning (all 200M parameters). Reference TSFM baseline: **Chronos-2** (`amazon/chronos-2`); statistical baselines: naive/persistence, seasonal-naive (week), moving average, ARIMA(1,1,0).

Data: the arena's **12 real yfinance series** (2023-08 → 2026-08 daily closes) plus the 48 real task datasets built from them (`*_dir5`, `*_volreg`, `*_extreme`, `*_trade`). Synthetic datasets and the empty `priv_*` stubs are excluded.

## 1. Forecasting (zero-shot)

Mean across the 12 real series of the per-series rolling-average metric (lower MASE / sMAPE / ND is better; ND = MAE relative to seasonal-naive).

| model | MASE h=1 | MASE h=5 | MASE h=12 | MASE h=21 | MASE h=63 |
|---|---|---|---|---|---|
| naive | 1.281 | 2.017 | 3.042 | 4.553 | 8.300 |
| seasonal_naive | 2.854 | 2.877 | 3.681 | 5.178 | 8.243 |
| moving_average | 3.416 | 3.878 | 4.555 | 5.317 | 8.400 |
| arima | 1.285 | 2.028 | 3.120 | 4.742 | 8.198 |
| chronos | 1.343 | 2.171 | 3.259 | 4.655 | 8.230 |
| timesfm@512 | 1.309 | 2.139 | 3.079 | 4.644 | 8.019 |
| timesfm@256 | 1.325 | 2.211 | 3.210 | 4.555 | 8.156 |
| timesfm@128 | 1.348 | 2.205 | 3.248 | 4.734 | 7.157 |

Dispersion at the primary horizon h=21 (across the 12 series / mean across-window sd):

| model | mean MASE | sd across series | mean across-window sd |
|---|---|---|---|
| naive | 4.553 | 2.446 | 2.300 |
| seasonal_naive | 5.178 | 2.633 | 2.910 |
| moving_average | 5.317 | 2.537 | 3.020 |
| arima | 4.742 | 2.558 | 2.287 |
| chronos | 4.655 | 2.480 | 2.562 |
| timesfm@512 | 4.644 | 2.448 | 2.596 |
| timesfm@256 | 4.555 | 2.576 | 2.310 |
| timesfm@128 | 4.734 | 2.697 | 2.424 |

### Per-asset MASE at the primary horizon h=21

| series | naive | seasonal_naive | moving_average | arima | chronos | timesfm@512 | timesfm@256 | timesfm@128 |
|---|---|---|---|---|---|---|---|---|
| `sp500_daily` (index) | 6.210 ± 3.453 | 7.068 ± 4.384 | 6.811 ± 4.144 | 6.351 ± 3.035 | 6.523 ± 3.353 | 6.333 ± 3.292 | 5.798 ± 2.634 | 6.191 ± 2.681 |
| `nasdaq_daily` (index) | 7.012 ± 3.322 | 7.892 ± 4.525 | 7.508 ± 4.735 | 7.165 ± 2.827 | 7.021 ± 3.734 | 7.266 ± 3.511 | 5.976 ± 2.645 | 7.391 ± 3.002 |
| `dow_daily` (index) | 5.834 ± 2.406 | 5.958 ± 3.596 | 6.918 ± 2.841 | 5.761 ± 2.554 | 5.788 ± 2.957 | 5.565 ± 2.560 | 6.165 ± 2.096 | 6.119 ± 2.281 |
| `russell2000_daily` (index) | 4.816 ± 2.334 | 5.347 ± 3.212 | 5.633 ± 2.996 | 4.736 ± 2.311 | 4.609 ± 2.381 | 4.816 ± 2.283 | 5.187 ± 2.858 | 4.833 ± 2.493 |
| `btc_crypto` (crypto) | 2.683 ± 1.888 | 2.865 ± 2.056 | 3.670 ± 2.643 | 2.666 ± 1.655 | 2.867 ± 2.137 | 3.676 ± 3.526 | 2.707 ± 1.969 | 2.839 ± 2.289 |
| `eth_crypto` (crypto) | 1.801 ± 1.001 | 1.867 ± 1.012 | 2.475 ± 1.398 | 1.822 ± 0.929 | 1.839 ± 1.027 | 1.765 ± 1.063 | 1.899 ± 0.998 | 2.078 ± 1.274 |
| `eur_usd_fx` (fx) | 2.312 ± 1.218 | 3.295 ± 1.657 | 4.079 ± 1.949 | 2.624 ± 1.228 | 2.728 ± 1.321 | 2.663 ± 1.476 | 2.881 ± 0.917 | 2.581 ± 1.126 |
| `gbp_usd_fx` (fx) | 2.763 ± 0.907 | 3.358 ± 0.987 | 3.058 ± 1.271 | 2.854 ± 0.975 | 2.802 ± 0.882 | 2.633 ± 0.955 | 2.670 ± 1.773 | 2.213 ± 1.078 |
| `gold_commodity` (commodity) | 6.001 ± 3.882 | 7.363 ± 4.456 | 7.309 ± 3.942 | 7.254 ± 4.237 | 6.821 ± 4.396 | 6.006 ± 4.427 | 6.625 ± 4.149 | 6.805 ± 4.299 |
| `crude_oil_commodity` (commodity) | 10.152 ± 4.152 | 11.033 ± 5.603 | 10.959 ± 7.953 | 10.442 ± 4.132 | 10.105 ± 5.639 | 10.255 ± 5.237 | 10.778 ± 5.623 | 10.976 ± 5.815 |
| `us_treasury_7y` (bond) | 2.106 ± 1.166 | 2.680 ± 1.514 | 2.185 ± 0.919 | 2.220 ± 1.378 | 1.937 ± 1.001 | 1.892 ± 0.935 | 1.860 ± 0.975 | 2.008 ± 0.913 |
| `ig_corp_bond` (bond) | 2.946 ± 1.870 | 3.405 ± 1.922 | 3.198 ± 1.452 | 3.008 ± 2.186 | 2.817 ± 1.913 | 2.852 ± 1.884 | 2.117 ± 1.087 | 2.777 ± 1.831 |

*Cell = mean ± sd of MASE over the non-overlapping rolling windows (the JSON also stores sd for every metric as `*_std`).*

### Normalized error vs the persistence forecast on identical windows (NDp = MAE(model)/MAE(naive); <1 means it beats 'keep the last price')

| model | NDp h=1 | NDp h=5 | NDp h=12 | NDp h=21 | NDp h=63 |
|---|---|---|---|---|---|
| naive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seasonal_naive | 2.179 | 1.412 | 1.219 | 1.157 | 1.010 |
| moving_average | 2.614 | 1.912 | 1.506 | 1.213 | 1.097 |
| arima | 1.002 | 1.007 | 1.028 | 1.042 | 0.975 |
| chronos | 1.043 | 1.073 | 1.041 | 1.024 | 0.965 |
| timesfm@512 | 1.015 | 1.046 | 1.008 | 1.029 | 0.961 |
| timesfm@256 | 1.024 | 1.088 | 1.057 | 0.997 | 0.975 |
| timesfm@128 | 1.046 | 1.083 | 1.067 | 1.029 | 0.880 |

### Context ablation (TimesFM zero-shot, mean over 12 series)

| horizon | ctx128 | ctx256 | ctx512 | ND@512 | CRPS@512 | 10% band violation (target 0.10) |
|---|---|---|---|---|---|---|
| h=1 | 1.348 | 1.325 | 1.309 | 1.120 | 118.59 | 0.092 |
| h=5 | 2.205 | 2.211 | 2.139 | 0.481 | 200.26 | 0.118 |
| h=12 | 3.248 | 3.210 | 3.079 | 0.526 | 315.67 | 0.134 |
| h=21 | 4.734 | 4.555 | 4.644 | 0.591 | 543.30 | 0.177 |
| h=63 | 7.157 | 8.156 | 8.019 | 1.207 | 1,568.3 | 0.105 |

## 2. Full fine-tuning (few-shot ladder → full train split)

All-parameter fine-tuning of the 200M module, lr=1e-05, context=256, target = the model's own 128-step output patch, loss = pinball over levels 0.1–0.9 + L1 on the median channel. Training windows come from the **train split only**. Zero-shot reference is the **matched-context** run (ctx=256).

| series | zero-shot ctx256 | FT K=16 | FT K=64 | FT full | FT final loss |
|---|---|---|---|---|---|
| `sp500_daily` | 5.798 | 5.764 | 5.826 | 5.820 | 0.389 |
| `nasdaq_daily` | 5.976 | 5.736 | 5.901 | 5.878 | 0.406 |
| `dow_daily` | 6.165 | 6.356 | 6.312 | 6.320 | 0.452 |
| `russell2000_daily` | 5.187 | 5.292 | 5.279 | 5.323 | 0.397 |
| `btc_crypto` | 2.707 | 2.894 | 3.102 | 3.147 | 0.365 |
| `eth_crypto` | 1.899 | 2.138 | 2.422 | 2.444 | 0.344 |
| `eur_usd_fx` | 2.881 | 2.676 | 2.591 | 2.511 | 0.862 |
| `gbp_usd_fx` | 2.670 | 2.701 | 2.813 | 2.652 | 0.736 |
| `gold_commodity` | 6.625 | 6.787 | 6.407 | 6.610 | 0.361 |
| `crude_oil_commodity` | 10.778 | 10.734 | 10.646 | 10.642 | 0.475 |
| `us_treasury_7y` | 1.860 | 1.820 | 1.818 | 1.844 | 0.399 |
| `ig_corp_bond` | 2.117 | 2.313 | 2.396 | 2.361 | 0.476 |

Mean MASE over the 12 series at h ∈ {5, 21}:

| setting | MASE | Δ vs zero-shot |
|---|---|---|
| zero-shot (matched ctx=256) | 3.383 | — |
| full fine-tune, K=fs16 | 3.410 | 0.027 |
| full fine-tune, K=fs64 | 3.422 | 0.039 |
| full fine-tune, K=full | 3.423 | 0.040 |

## 3. Classification on real labels

`*_dir5`: sign of the next 5-business-day return; `*_volreg`: tercile of the next 21-day realized vol (cut-points fitted on train). Macro-F1 (mean over the 12 series).

### dir5

| method | macro-F1 | accuracy | n series |
|---|---|---|---|
| majority | 0.342 | 0.524 | 12 |
| stat_logistic | 0.418 | 0.490 | 12 |
| stat_gbm | 0.523 | 0.562 | 12 |
| timesfm_forecastfeat_logistic | 0.440 | 0.504 | 12 |
| timesfm_forecastfeat_gbm | 0.483 | 0.507 | 12 |
| timesfm_probe_logistic | 0.444 | 0.522 | 12 |
| timesfm_probe_gbm | 0.424 | 0.484 | 12 |

### volreg

| method | macro-F1 | accuracy | n series |
|---|---|---|---|
| majority | 0.165 | 0.343 | 12 |
| stat_logistic | 0.240 | 0.331 | 12 |
| stat_gbm | 0.141 | 0.307 | 12 |
| timesfm_forecastfeat_logistic | 0.247 | 0.371 | 12 |
| timesfm_forecastfeat_gbm | 0.187 | 0.355 | 12 |
| timesfm_probe_logistic | 0.170 | 0.286 | 12 |
| timesfm_probe_gbm | 0.159 | 0.319 | 12 |

## 4. Extreme-move anomaly detection (real labels, expanding MAD, no look-ahead)

| method | macro-F1 | AUC-ROC | recall | n series |
|---|---|---|---|---|
| oracle_label_rule | 0.825 | 0.958 | 0.885 | 12 |
| stat_zscore | 0.495 | 0.956 | 0.771 | 12 |
| timesfm_bandviolation | 0.304 | 0.925 | 0.757 | 12 |
| timesfm_probe | 0.012 | 0.625 | 0.515 | 11 |

`oracle_label_rule` reproduces the label rule itself and is reported only as a reference point, not as a competitor; the fair comparisons are `stat_zscore` (rolling mean/sigma rule, threshold fitted on train) and the TSFM detectors. `n series` < 12 means some series have no extreme event in the evaluation span.


## 5. Quant-trading task (spec §2.3)

Signal = predicted h-day log return from the TSFM point forecast; positions `sign` (±1) and `voltarget` (clip(s/σ̂,−1,1)); non-overlapping 5-day holding periods on the test span; costs 0/10/30 bps one-way on |Δposition|.

| series | TSFM sign | TSFM vol-target | buy&hold | momentum | IC | hit rate | turnover |
|---|---|---|---|---|---|---|---|
| `sp500_daily` | -4.34 | -4.40 | 5.46 | -2.93 | -0.085 | 0.409 | 1.14 |
| `nasdaq_daily` | -1.93 | 2.07 | 3.33 | -2.79 | 0.092 | 0.409 | 0.59 |
| `dow_daily` | -8.62 | -8.86 | 8.15 | 6.66 | -0.177 | 0.273 | 0.86 |
| `russell2000_daily` | -7.47 | -3.36 | 7.91 | -3.63 | 0.047 | 0.318 | 0.95 |
| `btc_crypto` | 3.96 | 5.16 | -1.05 | 0.79 | 0.436 | 0.576 | 0.82 |
| `eth_crypto` | 0.33 | 3.22 | -1.06 | 6.47 | 0.261 | 0.545 | 0.94 |
| `eur_usd_fx` | 0.37 | 3.12 | 1.47 | -9.89 | 0.384 | 0.522 | 0.39 |
| `gbp_usd_fx` | 0.89 | 1.55 | 1.33 | -6.67 | 0.261 | 0.609 | 0.65 |
| `gold_commodity` | -0.83 | -0.27 | -4.84 | -1.68 | 0.022 | 0.545 | 1.50 |
| `crude_oil_commodity` | -1.30 | -2.35 | 1.29 | -0.05 | -0.171 | 0.545 | 0.05 |
| `us_treasury_7y` | 1.95 | 1.78 | -0.60 | -8.19 | 0.270 | 0.591 | 0.32 |
| `ig_corp_bond` | 2.73 | 5.93 | -1.23 | -5.81 | 0.542 | 0.591 | 0.50 |

Cost sensitivity (mean net Sharpe over the 12 series):

| strategy | 0 bps | 10 bps | 30 bps |
|---|---|---|---|
| TSFM sign | -0.42 | -1.19 | -2.69 |
| TSFM vol-target | 1.34 | 0.30 | -1.71 |
| TSFM FT(K=16) sign | 0.07 | -0.73 | -2.31 |
| TSFM FT(K=64) sign | -0.40 | -1.25 | -2.86 |
| TSFM FT(full) sign | -0.26 | -1.04 | -2.58 |
| TSFM FT(K=64) vol-target | 0.87 | -0.18 | -2.23 |
| buy & hold | 1.74 | 1.68 | 1.56 |
| momentum | -0.88 | -2.31 | -4.90 |

## 6. Protocol checks and caveats

- Test-span coverage of the rolling windows: mean **0.938**, min 0.808; no-leakage assertion held for every window set.
- MASE/ND scale uses the train split only; contexts always end strictly before the truth block, so no future information enters a forecast.
- Predictive-band calibration is reported as the 10% quantile violation rate (target 0.10); a TSFM whose bands are too narrow is misleading for risk use even when its point forecast is competitive.
- Trading significance: one asset at one horizon is breadth = 1 (Grinold), and this grid tests many configurations; per-strategy t-statistics are in the JSON (`t_stat_net`) and should be read with the multiple-testing caveat (Harvey 2017 / López de Prado 2018). No borrow cost, slippage, or funding spread is modelled; |position| ≤ 1, single asset.
- Fine-tuning regime: three years of a single daily series is a very small, very noisy corpus for a 200M-parameter model; the K-ladder and LR pilot in §2 quantify what full fine-tuning costs in out-of-sample accuracy.
