#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ft_%x_%A_%a.log
#SBATCH --error=logs/ft_%x_%A_%a.log
# Fine-tune one (MODEL, budget) across the series array.  Resubmit-safe: an existing
# checkpoint is skipped by the trainer, so re-running an array costs nothing.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

LONG12=(sp500_long nasdaq_long dow_long russell2000_long btc_long eth_long eurusd_long
        gbpusd_long gold_long crude_oil_long us_treasury_7y_long ig_corp_bond_long)
CORE12=(sp500_daily nasdaq_daily dow_daily russell2000_daily btc_crypto eth_crypto eurusd_fx
        gbp_usd_fx gold_commodity crude_oil_commodity us_treasury_7y ig_corp_bond)

if [ "${BUDGET:-full}" = "fs16" ]; then
  SERIES=("${LONG12[@]}" "${CORE12[@]}")   # 40 steps is cheap: cover every series
else
  SERIES=("${LONG12[@]}")                  # fs64/full only on the long histories
fi

i=${SLURM_ARRAY_TASK_ID:-0}
export MODEL BUDGET GPU_NAME=$(hostname)
export SERIES=${SERIES[$i]}
bash slurm/train_task.sh
