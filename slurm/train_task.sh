#!/usr/bin/env bash
# One fine-tuning task = one (model, series, budget).  Called by train.sbatch.
# Interpreter + memory knobs come from the family table (A6000 48GB, so batches are
# much larger than the 6GB laptop run); the trainer itself is family-specific and
# lives in its own uv env (envs/<family>/.venv), never in the driver's env.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

MODEL=${MODEL:?}  SERIES=${SERIES:?}  BUDGET=${BUDGET:-full}
CTX=${CTX:-512}   HORIZON=${HORIZON:-21}

export HF_HOME=${HF_HOME:-/home/lujun.li/hf_cache}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset TSFM_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # this box HAS network

env_for() {
  case "$1" in
    timesfm-2.5-200m)                      echo legacy ;;
    chronos-2|chronos-t5-base|chronos-t5-large) echo core ;;
    ttm-r2)                                echo ttm ;;
    moirai-*)                              echo moirai ;;
    toto-*)                                echo toto ;;
    moment-*)                              echo moment ;;
    kronos-*)                              echo kronos ;;
    tirex|tirex-2)                         echo tirex ;;
    timemoe-*)                             echo timemoe ;;
    *) echo core ;;
  esac
}
py_for() {
  local e; e=$(env_for "$1")
  for c in "envs/$e/.venv/bin/python" ".venv-$e/bin/python" ".venv2/bin/python" ".venv/bin/python"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo .venv/bin/python
}
knobs_for() {                       # big-GPU defaults; the trainer retries smaller on OOM
  case "$1" in
    timesfm-2.5-200m)  echo "--bs 32 --ctx 512 --amp bf16" ;;
    chronos-2)         echo "--bs 32 --ctx 512 --amp bf16" ;;
    chronos-t5-large)  echo "--bs 16 --ctx 512 --amp bf16" ;;
    ttm-r2)            echo "--bs 64 --ctx 512 --amp fp32" ;;
    moirai-2.0-R-small) echo "--bs 32 --ctx 512 --amp bf16" ;;
    moirai-1.1-R-base) echo "--bs 16 --ctx 512 --amp bf16" ;;
    toto-2-22m)        echo "--bs 32 --ctx 512 --amp bf16" ;;
    toto-2-313m)       echo "--bs 8  --ctx 256 --amp bf16" ;;
    moment-1-large)    echo "--bs 16 --ctx 512 --amp fp32" ;;
    kronos-small)      echo "--bs 16 --ctx 512 --run-training" ;;
    kronos-base)       echo "--bs 8  --ctx 512 --run-training" ;;
    *)                 echo "--bs 16 --ctx 512 --amp bf16" ;;
  esac
}

PY=$(py_for "$MODEL")
KNOBS=$(knobs_for "$MODEL")
echo "== [$(date -Is)] $MODEL / $SERIES / $BUDGET on $GPU_NAME using $PY $KNOBS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

$PY -m tsfm4finance.bench.trainers --model "$MODEL" --series "$SERIES" \
    --budget "$BUDGET" --horizon "$HORIZON" $KNOBS --device cuda
rc=$?
echo "== [$(date -Is)] $MODEL / $SERIES / $BUDGET rc=$rc"
exit $rc
