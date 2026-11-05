# TimesFM benchmark run — execution report (plan S1–S12)

This is the record of the executed benchmark behind `reports/02_timesfm_benchmark.md`
and §4.4 of the survey paper (`TSFM_Paper/chapters/4_datasets.tex`).

## 1. What was run

| stage | content | count |
|---|---|---|
| `tasks` | real task datasets built from the 12 real yfinance series (`*_dir5`, `*_volreg`, `*_extreme`, `*_trade`) | 48 datasets |
| `finetune` | full-parameter fine-tuning of TimesFM 2.5-200M per series: K=16, K=64, full train split | 36 runs |
| `forecast` | rolling-origin backtest, horizons h ∈ {1,5,12,21,63}, stride = h, context ≤ 512; 8 generic models + 3 fine-tuned variants per series | 660 rows |
| `classify` | `dir5` / `volreg` × {statistical, TSFM forecast-features, TSFM embedding probe} × {logistic, GBM} + majority | 24 datasets |
| `anomaly` | `extreme` × {oracle label rule, rolling mean/σ z-score, TSFM band violation, TSFM embedding probe} | 12 datasets |
| `trading` | sign / vol-target positions × costs {0,10,30} bps, zero-shot and 3 fine-tuned signal sets, buy&hold + momentum + random benchmarks | 12 datasets |
| `derive` | NDp = MAE(model)/MAE(persistence) on identical windows | 60 cells |
| `leaderboard` | appends `protocol: "spec-4.1-rolling"` rows at h=21, tags older rows `legacy-single-window` | 132 new rows |
| `report` | renders `reports/02_timesfm_benchmark.md` (means, ± sd over windows, ± sd over series) | — |

Status: **complete, zero error entries** — `finetune 36, forecast 660, classify 24,
anomaly 12, trading 12`; every forecast row carries `info.no_leakage: true`,
`stride == horizon`, `max_context: 512`, and mean test-span coverage **0.938**.
Machine check: `pytest tests/test_bench.py` (completion-invariant tests).

## 2. Headline results (mean over the 12 real series)

Forecasting (MASE, lower better; NDp = MAE vs persistence on the same windows):

| model | h=1 | h=5 | h=12 | h=21 | h=63 | NDp(h=21) |
|---|---|---|---|---|---|---|
| persistence | 1.281 | 2.017 | 3.042 | 4.553 | 8.300 | 1.000 |
| seasonal naive | 2.854 | 2.877 | 3.681 | 5.178 | 8.243 | 1.157 |
| ARIMA(1,1,0) | 1.285 | 2.028 | 3.120 | 4.742 | 8.198 | 1.042 |
| Chronos-2 | 1.343 | 2.171 | 3.259 | 4.655 | 8.230 | 1.024 |
| TimesFM 2.5 (zero-shot, ctx 512) | 1.309 | 2.139 | 3.079 | 4.644 | 8.019 | 1.029 |

* Full fine-tuning never improved out-of-sample MASE on average
  (ΔMASE over h∈{5,21} vs matched-context zero-shot: **+0.027** (K=16), **+0.039** (K=64),
  **+0.040** (full train split)) while the training loss fell from 2.08 to 0.41.
  Heterogeneous by asset class: helps fx (−0.109) and commodity (−0.017),
  hurts bond (+0.059), index (+0.028) and crypto (+0.251).
* Direction (`dir5`) macro-F1: statistical GBM **0.523** > TSFM routes **0.483** >
  majority 0.342. Vol regime (`volreg`): TSFM forecast-features **0.247** vs statistical **0.240**.
* Extreme moves: label-generating rule 0.825 (reference only), fair statistical z-score rule
  **0.495 F1 / 0.956 AUC**, TSFM band violation 0.304 F1 / **0.925 AUC** — the TSFM band
  *ranks* almost as well but its default threshold under-detects.
* Trading (net Sharpe at 10 bps, mean over series): TSFM sign **−1.19**, vol-target **0.30**,
  fine-tuned sign −0.73; buy & hold **1.68**, momentum −2.31; mean rank-IC 0.157.
  No configuration beats buy & hold net of costs on this 2023–2026 span.
* Calibration: 10% predictive band violated on 0.092 (h=1) → 0.177 (h=21) of outcomes.

## 3. Reproduce

```bash
cd ~/projects/TSFM4Finance
.venv/bin/python -m tsfm4finance.bench.run                 # all stages, incremental
.venv/bin/python -m tsfm4finance.bench.paper_table --apply # regenerate the paper §4.4 block
pytest tests/test_bench.py                                 # completion invariants
```

The run is incremental: finished `(dataset, horizon, model)` keys are skipped, so an
interrupted sweep resumes. Each 200M checkpoint is ~700 MB; the harness deletes a
checkpoint as soon as its forecast rows and trading rows exist (`_gc_checkpoints` in
`tsfm4finance/bench/run.py`), and `finetune` retrains only while some consumer still needs
the weights. On a disk-constrained machine run the sweep in prefix chunks
(`--limit 2/4/…/12`, `finetune` then `trading` per chunk) to keep ≤ 6 checkpoints on disk.

## 4. Known caveats (read before quoting numbers)

* Three years of daily data ⇒ few non-overlapping windows at long horizons
  (5–8 origins at h=21 for the shorter series); across-window sd is large and is reported.
* MASE is scaled by the *train-split* naive error, so absolute MASE > 1 on trending
  series is expected; NDp is the comparable column.
* Trading results are single-asset, breadth-one bets over one bull regime; with a
  12-series × 5-horizon × several-rule grid, Sharpe t-stats must be read with the
  multiple-testing deflators of the backtesting literature.
* Anomaly `oracle_label_rule` equals the label definition; it exists to expose how much
  of the task is definitional, not as a competitor.
