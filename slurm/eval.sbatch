#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_%x_%A.log
# Score the fine-tuned checkpoints: plan only what exists, then run every stage.
# Resumable per cell (reports/state.sqlite3), so re-submitting continues where it stopped.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=${HF_HOME:-/home/lujun.li/hf_cache} PYTHONUNBUFFERED=1
G=${GROUPS:-long12,core12,newx}   H=${HORIZONS:-5,21}   AD=${ADAPTATIONS:-zero,fs16,fs64,full}
envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage plan --tiers A,B --groups $G \
    --horizons $H --contexts 512 --adaptations $AD 2>&1 | tail -3
for S in forecast trading classify anomaly; do
  echo "==== $S $(date -Is)"
  envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage $S --tiers A,B --groups $G \
      --horizons $H --contexts 512 --adaptations $AD --workers 1 --device cuda 2>&1 | tail -4
done
envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage status 2>&1 | tail -12
envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage export 2>&1 | tail -3
echo EVAL_DONE
