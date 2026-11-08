# Fine-tuning on your GPU server (per-model trainers)

Every model family has **its own trainer**: the families disagree on scaling,
heads and which parameters may move, so there is no single generic loop.

```bash
cd TSFM4Finance
python -m tsfm4finance.bench.trainers --list          # trainer status per family
python -m tsfm4finance.bench.trainers --model <key> --series <dataset> --budget full
```

Use the python interpreter of the model's environment (see `bench/config.yaml`
→ `envs`); the CLI refuses to run a model in the wrong environment unless you
pass `--ignore-env`.

| family | trainer | how | smoke command |
|---|---|---|---|
| TimesFM 2.5 | `trainers/timesfm25.py` | full-parameter pinball+L1 loop (the one behind §4.4) | `.venv/bin/python -m tsfm4finance.bench.trainers --model timesfm-2.5-200m --series sp500_daily --budget fs16 --smoke` |
| Chronos-2 | `trainers/chronos2.py` | vendor `Chronos2Pipeline.fit(finetune_mode="full")` | `.venv2/bin/python ... --model chronos-2 --series sp500_daily --budget fs16 --smoke` |
| Moirai 1.1/2.0/MoE | `trainers/moirai.py` | pinball through the predictor forward | `.venv-moirai/bin/python ... --model moirai-2.0-R-small --budget fs16 --smoke` |
| Granite TTM | `trainers/ttm.py` | pinball on the model's quantile grid | `.venv-ttm/bin/python ... --model ttm-r2 --budget fs16 --smoke` |
| MOMENT-1 | `trainers/moment.py` | L1 on the forecasting head (+backbone) | `.venv-moment/bin/python ... --model moment-1-large --budget fs16 --smoke --bs 1` |
| Toto 2.0 | `trainers/toto.py` | pinball in the model's asinh-scaled space | `.venv-toto/bin/python ... --model toto-2-22m --budget fs16 --smoke` |
| Kronos | `trainers/kronos.py` | exports OHLCV to Kronos' own `finetune_csv` stack and runs (or prints) it | `.venv-kronos/bin/python ... --model kronos-small --series sp500_long --budget full` |
| TimesFM 3.0, Chronos-Bolt, Time-MoE, TiRex(-2), FlowState | **trainer pending** (reasons in `--list`) | – | – |

## Budgets (fixed by the benchmark definition)

| budget | meaning | steps | lr |
|---|---|---|---|
| `fs16` | 16 train windows | 40 | 1e-5 |
| `fs64` | 64 train windows | 120 | 1e-5 |
| `full` | every train window, 3 epochs | auto | 5e-6 |

Windows come **only from the train split**; each checkpoint writes a JSON sidecar
with `train_end`, the trainable-parameter count, the loss curve and the profile
(`device/amp/ctx/batch`) that actually ran — so a head-only run can never be
reported as a full fine-tune.

## Recommended server run (this is the part we did not run locally)

One script, resumable, offline (weights from `./models`, see `docs/08_offline_models.md`):

```bash
reports/run_train.sh fs16          # 16 windows / 40 steps  x all 24 series  (cheap, do it first)
reports/run_train.sh fs64          # 64 windows / 120 steps x core12
reports/run_train.sh full          # every train window, 3 epochs x long12
reports/run_train.sh fs16 --smoke              # 2 optimiser steps per family, nothing scored
reports/run_train.sh full --models chronos-2 --series sp500_long,btc_long --force
```
It picks the right interpreter and memory knobs per family, skips checkpoints that
already exist, retries once at `--bs 1 --ctx 128` if a run dies on VRAM, and logs to
`reports/train_<budget>.log`.  Do not run it while `reports/run_sweep_pass1.sh` owns
the same GPU (or pass `--device cuda:1`).

Equivalent explicit commands (what the script runs):

```bash
export TSFM_WEIGHTS_ROOT=$PWD/models TSFM_OFFLINE=1 HF_HOME=$PWD/.hf-cache
S24=sp500_daily,nasdaq_daily,dow_daily,russell2000_daily,btc_crypto,eth_crypto,eurusd_fx,gbp_usd_fx,gold_commodity,crude_oil_commodity,us_treasury_7y,ig_corp_bond
L12=sp500_long,nasdaq_long,dow_long,russell2000_long,btc_long,eth_long,eurusd_long,gbpusd_long,gold_long,crude_oil_long,us_treasury_7y_long,ig_corp_bond_long

.venv/bin/python        -m tsfm4finance.bench.trainers --model timesfm-2.5-200m --series $L12 --budget full --bs 4  --ctx 512 --amp fp32 --horizon 21
.venv2/bin/python       -m tsfm4finance.bench.trainers --model chronos-2        --series $L12 --budget full --bs 16 --ctx 512 --amp bf16 --horizon 21
.venv-ttm/bin/python    -m tsfm4finance.bench.trainers --model ttm-r2           --series $L12 --budget full --bs 32 --ctx 512 --amp fp32 --horizon 21
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-2.0-R-small --series $L12 --budget full --bs 8 --ctx 512 --amp bf16 --horizon 21
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-1.1-R-base  --series $L12 --budget full --bs 4 --ctx 512 --amp bf16 --horizon 21
.venv-toto/bin/python   -m tsfm4finance.bench.trainers --model toto-2-22m       --series $L12 --budget full --bs 8  --ctx 512 --amp bf16 --horizon 21
.venv-toto/bin/python   -m tsfm4finance.bench.trainers --model toto-2-313m      --series $L12 --budget full --bs 2  --ctx 256 --amp bf16 --horizon 21
.venv-moment/bin/python -m tsfm4finance.bench.trainers --model moment-1-large   --series $L12 --budget full --bs 2  --ctx 256 --amp fp32 --horizon 21
.venv-kronos/bin/python -m tsfm4finance.bench.trainers --model kronos-small     --series $L12 --budget full --bs 8  --ctx 256 --run-training --horizon 21
```
`--smoke` writes to `reports/ft_smoke/` and those checkpoints are ignored by the
planner and by scoring, so a 2-step wiring check can never end up in the paper.

Then re-plan and run the sweep: the planner only creates fine-tuned cells for
checkpoints that exist, so the new rows appear automatically.

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan --tiers A,B --groups core12,long12 --horizons 5,21
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --workers 1
.venv2/bin/python -m tsfm4finance.bench.sweep --stage gc          # drop consumed checkpoints
```

Memory knobs for small GPUs: `--amp bf16`, `--bs 1..4`, `--ctx 192/256`,
`--heads-only` (recorded as an ablation, not as a full fine-tune).
