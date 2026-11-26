#!/usr/bin/env bash
# First *scored* rows from the DGX Spark (gpu7: GB10, aarch64, unified memory).
#
# The audit proved torch sees CUDA there. That is not the same as the evaluator being
# able to produce a merged-quality row, and two things about this machine only show up
# when you try:
#   - nvidia-smi answers [N/A] for memory on GB10, so anything that sizes work from
#     free memory has to ask torch instead (see trux_lane_runner.parse_gpu_rows);
#   - there is no route to the data plane from here, so the corpus has to arrive over
#     the same ssh channel the worker already uses.
#
# Protocol is pass-3's exactly (ctx 256, horizons 21/63, 8 windows, seed 0) so these
# cells join the existing merge key instead of sitting beside it.
set -euo pipefail
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
REPO=${REPO:-}
for c in "$HOME/TSFM4Finance-trl/trl" "$HOME/TSFM4Finance-trl" "$HOME/tsfm-trl"; do
  [ -f "$c/scripts/sweep_evaluate.py" ] && { REPO=$c; break; }
done
[ -n "$REPO" ] || { echo "FATAL: no harness clone (looked for scripts/sweep_evaluate.py under \$HOME)"; exit 6; }

TRUX=${TRUX:-yewei@trux-tsubame.uni.lux}
TRUX_SSH=${TRUX_SSH:-"ssh -i $HOME/.ssh/id_ed25519_gm_transfer -o StrictHostKeyChecking=accept-new"}
TRUX_CSV=${TRUX_CSV:-/raid/yewei/projects_yewei/TSFM4Finance-trl/trl/reports/pass3_trux/csv}
CSVROOT=${CSVROOT:-$HOME/spark_bridge/csv}
OUT=${OUT:-$HOME/spark_bridge/outbox/results/gpu7_smoke.jsonl}
# names, not a directory copy: a smoke test should fail for the real reason instead of
# after transferring 15 MB of corpus it did not need
SERIES_FILES=${SERIES_FILES:-"std_equity_1d_10107.csv std_fx_1d_euraud.csv std_equity_1d_11308.csv"}

echo "host=$(hostname) repo=$REPO sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "date=$(date -u +%FT%TZ)"

echo "-- device probe (what the row will claim as its device)"
"$ENVROOT/core/bin/python" - <<'PY'
import torch
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  {i} {torch.cuda.get_device_name(i)} free={free >> 20}MiB total={total >> 20}MiB")
PY

echo "-- corpus: pulling ${SERIES_FILES// /, } from $TRUX"
mkdir -p "$CSVROOT"
for f in $SERIES_FILES; do
  rsync -e "$TRUX_SSH" -a "$TRUX_CSV/$f" "$CSVROOT/" || { echo "FATAL: corpus pull failed for $f"; exit 5; }
done
wc -l "$CSVROOT"/*.csv 2>/dev/null | sed 's/^/  /'

run() {
  local models=$1 out=$2; shift 2
  local args=()
  for f in $SERIES_FILES; do args+=(--series "$CSVROOT/$f"); done
  "$ENVROOT/core/bin/python" -u "$REPO/scripts/sweep_evaluate.py" \
    "${args[@]}" --models "$models" --horizons 21,63 --max-windows 8 --ctx 256 \
    --seed 0 --out "$out" --fresh "$@"
}

# baselines ride along so a CUDA-scored row exists even if the Hugging Face download is
# blocked on this machine; losing the TSFM cell is then visible, not silent
if run "naive,seasonal_naive,chronos-bolt-small" "$OUT"; then
  final=$OUT
else
  echo "WARN: weights run failed (weights blocked?); re-running baselines only"
  final=${OUT%.jsonl}_baselines.jsonl
  run "naive,seasonal_naive" "$final"
fi

echo "rows=$(wc -l < "$final")"
echo "-- first row (provenance is the point of this smoke):"
head -1 "$final" | cut -c1-500 | sed 's/^/  /'
cp "$final" "$HOME/spark_bridge/outbox/" 2>/dev/null || true
