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

```bash
# Tier-A models x the 12 long-history series, full fine-tune at h=21
for M in timesfm-2.5-200m chronos-2; do
  .venv2/bin/python -m tsfm4finance.bench.trainers --model $M \
      --series sp500_long,nasdaq_long,dow_long,russell2000_long,btc_long,eth_long,eurusd_long,gbpusd_long,gold_long,crude_oil_long,us_treasury_7y_long,ig_corp_bond_long \
      --budget full --epochs 3 --bs 8 --ctx 512 --amp bf16
done
.venv-moirai/bin/python -m tsfm4finance.bench.trainers --model moirai-2.0-R-small --series sp500_long --budget full --amp bf16
.venv-ttm/bin/python  -m tsfm4finance.bench.trainers --model ttm-r2 --series sp500_long --budget full
.venv-moment/bin/python -m tsfm4finance.bench.trainers --model moment-1-large --series sp500_long --budget full --bs 2
.venv-toto/bin/python -m tsfm4finance.bench.trainers --model toto-2-22m --series sp500_long --budget full
.venv-kronos/bin/python -m tsfm4finance.bench.trainers --model kronos-small --series sp500_long --budget full --run-training
```

Then re-plan and run the sweep: the planner only creates fine-tuned cells for
checkpoints that exist, so the new rows appear automatically.

```bash
.venv2/bin/python -m tsfm4finance.bench.sweep --stage plan --tiers A,B --groups core12,long12 --horizons 5,21
.venv2/bin/python -m tsfm4finance.bench.sweep --stage forecast --workers 1
.venv2/bin/python -m tsfm4finance.bench.sweep --stage gc          # drop consumed checkpoints
```

Memory knobs for small GPUs: `--amp bf16`, `--bs 1..4`, `--ctx 192/256`,
`--heads-only` (recorded as an ablation, not as a full fine-tune).
