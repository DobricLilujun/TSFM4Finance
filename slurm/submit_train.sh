#!/usr/bin/env bash
# Submit the fine-tuning fleet to Slurm (watson: partition gpu, 3x RTX A6000 48GB).
#
#   slurm/submit_train.sh fs16              # cheap few-shot ladder, all 24 series
#   slurm/submit_train.sh fs64              # long histories only
#   slurm/submit_train.sh full              # every train window, 3 epochs
#   slurm/submit_train.sh full chronos-2    # one family only
#   DRY=1 slurm/submit_train.sh full        # print what would be submitted
#
# One array per model, one task per series -> Slurm spreads them over the GPUs as
# they free up, and one series failing does not take the model's whole run down.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

BUDGET=${1:-full}
ONLY=${2:-}
shift || true; shift || true

case $BUDGET in
  fs16) T=00:30:00; NS=24 ;;
  fs64) T=01:30:00; NS=12 ;;
  full) T=06:00:00; NS=12 ;;
  *) echo "budget must be fs16|fs64|full"; exit 2 ;;
esac

# every family with a real trainer (bench.trainers --list); families without one are
# evaluated zero-shot only, deliberately not queued here
MODELS=(timesfm-2.5-200m chronos-2 ttm-r2 moirai-2.0-R-small moirai-1.1-R-base
        toto-2-22m toto-2-313m moment-1-large kronos-small kronos-base)
if [ -n "$ONLY" ]; then MODELS=("$ONLY"); fi

for M in "${MODELS[@]}"; do
  CMD=(sbatch --job-name="ft-${M:0:14}-${BUDGET}" --time="$T"
         --array="0-$((NS-1))" --mem=24G --export=ALL,MODEL=$M,BUDGET=$BUDGET
         slurm/train.sbatch)
  if [ "${DRY:-0}" = "1" ]; then echo "${CMD[*]}"; else
    printf "%-20s %-5s -> " "$M" "$BUDGET"; "${CMD[@]}"
  fi
done

echo
echo "squeue -u $USER   # watch the fleet"
