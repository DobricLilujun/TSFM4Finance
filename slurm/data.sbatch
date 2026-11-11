#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/data_%x_%A.log
# CPU-only: fill in the dataset tiers (t2 cross-sectional panel + external benchmarks,
# t3 intraday crypto bars, t4 Scrapling hunt) and re-derive the task datasets.
# No --gres: it packs into the idle CPUs of a gpu node instead of waiting for a GPU.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=${HF_HOME:-/home/lujun.li/hf_cache} PYTHONUNBUFFERED=1
envs/core/.venv/bin/python -m tsfm4finance.bench.download --tier ${TIERS:-t2,t3,t4} 2>&1 | tail -20
envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage derive --groups long12,core12,newx,fred 2>&1 | tail -4
envs/core/.venv/bin/python -m tsfm4finance.bench.download --status 2>&1 | tail -12
echo DATA_STAGE_DONE
