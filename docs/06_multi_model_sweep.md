# Multi-model sweep: install matrix, datasets, resumable runs

## Why several python environments

| env | python | models | note |
|---|---|---|---|
| `.venv2` (core) | 3.11 | TimesFM 3.0, Chronos-2/Bolt/T5, baselines, statsmodels/pmdarima | sweep driver lives here |
| `.venv` (legacy) | 3.11 | TimesFM 2.5-200M | the env that produced the published §4.4 numbers |
| `.venv-moirai` | 3.11 | Moirai 1.1 / 2.0 / Moirai-MoE | `uni2ts` pins torch<2.5 |
| `.venv-moment` | 3.11 | MOMENT-1-large | `momentfm` pins numpy 1.25, `huggingface-hub<1.0` |
| `.venv-ttm` | 3.11 | Granite TTM r2, FlowState r1 | `tsfm_public` |
| `.venv-toto` | 3.12 | Toto 2.0 (22m/313m/1B/2.5B) | `toto-2` needs python>=3.12 |
| `.venv-tirex` | 3.11 | TiRex | xLSTM kernels |
| `.venv-tirex2` | 3.12 | TiRex-2 | **kernel-blocked locally**: `mlstm_kernels`/`flashrnn` Triton sources use `tl.make_block_ptr`, removed in Triton 3.9 (torch 2.15/CUDA 13.2). On the server use torch 2.6-2.8 + triton 3.2-3.4 |
| `.venv-kronos` | 3.11 | Kronos small/base (+ its `finetune_csv` trainer) | needs OHLCV panels |
| `.venv-timemoe` | 3.11 | Time-MoE 50m/200m | remote code needs `transformers==4.44` |

Create them with:

```bash
uv venv .venv2 --python 3.11 && uv pip install --python .venv2/bin/python -e ".[bench]" timesfm chronos-forecasting pmdarima statsmodels
uv venv .venv-toto --python 3.12 && uv pip install --python .venv-toto/bin/python "toto-2" pyyaml requests huggingface_hub pyarrow
# ... one per row above; the module names are in bench/config.yaml -> envs
```

## Data tiers (`--stage prefetch-data`)

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage prefetch-data --tiers t1,t2,t3,t4
.venv2/bin/python -m tsfm4finance.bench.download --status
```

| tier | what | status |
|---|---|---|
| T1 | long-history OHLCV panel (2004→) for the 12 arena series + ~32 new instruments (VIX, rates, FX, commodities, global indices, sector ETFs) + FRED macro panel (31 series, `fredgraph.csv`, no API key) | downloaded (156 datasets on disk) |
| T2 | cross-sectional Nasdaq-100 close/return panel + external benchmarks (fev-bench `fred_md_2025`, `fred_qd_2025`, `us_consumption`; GIFT-Eval `m4_daily/weekly/hourly`) | partially downloaded; re-run to retry failures |
| T3 | Binance hourly crypto bars from 2019 | command ready; not run locally (bandwidth) |
| T4 | Scrapling hunt for licensed/keyed sources | reports *what needs your credentials*; nothing is silently skipped |

Derived task datasets (classification / anomaly / trading) are built locally:

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage derive --groups core12,long12
```

## Resumable sweep

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan --tiers A,B --groups core12,long12 --horizons 5,21
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --workers 1     # or:
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --env core --shard 1 --n-shards 4   # 4 machines/workers
.venv2/bin/python -m tsfm4finance.bench.sweep --stage status                    # writes reports/PROGRESS.md
.venv2/bin/python -m tsfm4finance.bench.sweep --stage export                    # reports/multi_model_results.json
```

Cells live in `reports/state.sqlite3` (WAL). Each cell is
`(stage, series, model, adaptation, horizon, span, seed)`; artifacts are one JSON
file per cell. Interrupts, `kill -9` and reboots are safe: `running` rows are
re-claimed on the next start, `done` rows are never recomputed, and
`skipped(reason)` / `failed(error)` rows stay visible in `PROGRESS.md` — a gap in
the paper is always traceable to a row.

Degradation ladder (per cell, recorded in the `profile` column):
`fp32 → bf16 → fp16 → ctx/2,bs/2 → ctx/4 → cpu → skipped(oom)`.

## Engine status (h=21, `sp500_long`, 4 windows, `reports/smoke_all.log`)

| model | ok | note |
|---|---|---|
| naive / seasonal_naive / moving_average / arima / arima_auto / ets / theta | yes | ARIMA(1,1) often collapses to persistence on daily indices |
| chronos-2, chronos-bolt-base, chronos-t5-* | yes | |
| timesfm-2.5-200m, timesfm-3.0 | yes | |
| moirai-2.0-R-small, moirai-1.1-R-base | yes | quantile grid interpolated to the benchmark levels |
| toto-2-22m | yes | best zero-shot MASE in the probe set |
| ttm-r2, flowstate-r1 | yes | TTM has a fixed prediction length; no quantiles on r2 |
| tirex | yes | 32-step output patch → median rollout for longer horizons |
| kronos-small / kronos-base | yes | consumes OHLCV candle windows |
| moment-1-large | yes | point forecast + embeddings + anomaly reconstruction |
| timemoe-50m/200m | blocked in `.venv2` | needs `transformers==4.44` (`.venv-timemoe`) |
| tirex-2 | blocked locally | Triton/CUDA kernel incompatibility; server recipe above |
