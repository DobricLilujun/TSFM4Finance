#!/usr/bin/env bash
# Fine-tuning ladder for the multi-model benchmark - one trainer per model family,
# resumable, offline (weights come from ./models, nothing is downloaded here).
#
#   reports/run_train.sh fs16              # 16 windows / 40 steps, all 24 series
#   reports/run_train.sh fs64              # 64 windows / 120 steps, core12
#   reports/run_train.sh full              # every train window, 3 epochs, long12
#   reports/run_train.sh fs16 --smoke      # 2 optimiser steps: proves the loop is wired
#   reports/run_train.sh fs16 --models chronos-2,ttm-r2
#   reports/run_train.sh full  --series sp500_long,btc_long
#
# Existing checkpoints are skipped (delete them or pass --force to retrain), so an
# interrupted run just re-runs the same line.  A checkpoint belongs to one series;
# the evaluator refuses to score series B with a checkpoint trained on series A.
#
# Do NOT run this at the same time as reports/run_sweep_pass1.sh on the same GPU:
# either wait for the sweep, or pass --device cuda:1 for the second job.
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------- offline weights ----------------------------------------------------
export TSFM_WEIGHTS_ROOT=${TSFM_WEIGHTS_ROOT:-$PWD/models}
export TSFM_OFFLINE=${TSFM_OFFLINE:-1}
[ -d .hf-cache ] && export HF_HOME=${HF_HOME:-$PWD/.hf-cache}

BUDGET=${1:-fs16}; shift || true
SMOKE=""; MODELS=""; SERIES=""; EXTRA=""
while [ $# -gt 0 ]; do
  case $1 in
    --smoke)  SMOKE="--smoke"; shift ;;
    --force)  EXTRA="$EXTRA --force"; shift ;;
    --device) EXTRA="$EXTRA --device $2"; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    --series) SERIES=$2; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

CORE12=sp500_daily,nasdaq_daily,dow_daily,russell2000_daily,btc_crypto,eth_crypto,eurusd_fx,gbp_usd_fx,gold_commodity,crude_oil_commodity,us_treasury_7y,ig_corp_bond
LONG12=sp500_long,nasdaq_long,dow_long,russell2000_long,btc_long,eth_long,eurusd_long,gbpusd_long,gold_long,crude_oil_long,us_treasury_7y_long,ig_corp_bond_long
case $BUDGET in
  fs16) SERIES=${SERIES:-$CORE12,$LONG12} ;;   # cheap: cover every series
  fs64) SERIES=${SERIES:-$CORE12} ;;
  full) SERIES=${SERIES:-$LONG12} ;;           # long history = the honest full fine-tune
  *) echo "budget must be fs16|fs64|full"; exit 2 ;;
esac

# ---------- model -> interpreter + memory knobs --------------------------------
# family trainers live in envs that cannot be merged (see docs/09_uv_environments.md);
# knobs are what actually fit / what the published TimesFM run used.
MODELS_TSV=$(cat <<'TSV'
timesfm-2.5-200m    .venv/bin/python       --bs 4  --ctx 512 --amp fp32
chronos-2           .venv2/bin/python      --bs 16 --ctx 512 --amp bf16
ttm-r2              .venv-ttm/bin/python   --bs 32 --ctx 512 --amp fp32
moirai-2.0-R-small  .venv-moirai/bin/python --bs 8  --ctx 512 --amp bf16
moirai-1.1-R-base   .venv-moirai/bin/python --bs 4  --ctx 512 --amp bf16
toto-2-22m          .venv-toto/bin/python  --bs 8   --ctx 512 --amp bf16
toto-2-313m         .venv-toto/bin/python  --bs 2   --ctx 256 --amp bf16
moment-1-large      .venv-moment/bin/python --bs 2  --ctx 256 --amp fp32
chronos-t5-large    .venv2/bin/python      --bs 4   --ctx 512 --amp bf16
kronos-small        .venv-kronos/bin/python --bs 8  --ctx 256 --amp fp32 --run-training
kronos-base         .venv-kronos/bin/python --bs 4  --ctx 256 --amp fp32 --run-training
TSV
)

if [ -n "$MODELS" ]; then
  MODELS_TSV=$(echo "$MODELS_TSV" | grep -E "^[[:space:]]*($(echo "$MODELS" | sed 's/,/|/g'))[[:space:]]")
fi

LOG=reports/train_${BUDGET}${SMOKE:+_smoke}.log
echo "== training budget=$BUDGET series=$(echo $SERIES | tr ',' ' ' | wc -w) log=$LOG" | tee -a "$LOG"
FAIL=0
while read -r M PY KNOBS; do
  [ -z "${M:-}" ] && continue
  echo "-- $M  [$PY] $KNOBS"
  if ! $PY -m tsfm4finance.bench.trainers --model "$M" --series "$SERIES" \
        --budget "$BUDGET" --horizon 21 $KNOBS $SMOKE $EXTRA >>"$LOG" 2>&1; then
    # retry once at minimum memory: on a shared/small GPU the first attempt often
    # dies on fragmentation or on another process holding VRAM
    echo "   retrying $M at --bs 1 --ctx 128" | tee -a "$LOG"
    if ! $PY -m tsfm4finance.bench.trainers --model "$M" --series "$SERIES" \
          --budget "$BUDGET" --horizon 21 $KNOBS $SMOKE $EXTRA --bs 1 --ctx 128 >>"$LOG" 2>&1; then
      echo "   !! $M failed (see $LOG)" | tee -a "$LOG"; FAIL=$((FAIL+1))
    fi
  fi
done <<< "$MODELS_TSV"

echo
echo "== checkpoints under reports/ft_ckpts for budget=$BUDGET"
ls reports/ft_ckpts 2>/dev/null | grep -E "__${BUDGET}(\.pt|_pipe|_dir|\.json|$)" | sort | sed 's/^/   /' | nl
echo "== failures: $FAIL   (log: $LOG)"
