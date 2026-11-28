#!/usr/bin/env bash
# gpu7 (DGX Spark, GB10) pass 7: the same cells at 32 and 64 windows, for the window-sample question.
#
# Why: every published number so far sits at --max-windows 8, and docs/08 measures that the window
# sample is part of the protocol - at 8 windows, seed 0 vs seed 1 moves naive's own MASE ~55% median
# and the paired model/naive ratio ~12%. So the ties in the ranking (chronos-bolt-small vs chronos-2
# at 61.6% pairwise, chronos-bolt-base vs chronos-t5-small at 53.8%) could be an artifact of drawing
# only 8 windows. This lane re-measures the tie cluster - plus naive/seasonal_naive/arima/theta so the
# beat-naive share is computable at the new window count - at 32 and 64 windows, seed 0, same series,
# same bytes. `n_windows` is in the protocol signature, so these rows never merge with the 8-window
# rows: the comparison is done per window count by probes/window_sensitivity.py.
#
# GENERATED before dispatch - one file crosses the bridge:
#
#   probes/trux/bridge/gpu7_generate.sh \
#       probes/trux/bridge/gpu7_060_windows.sh probes/trux/gpu7_full66.series /tmp/job.sh
#   probes/trux/bridge/trux_dispatch.sh /tmp/job.sh
#
# Machine-specific endpoints (host, user, path, key) come from an untracked sibling
# file so this public tree carries no lab hostnames or internal IPs; the example
# next to it lists the variables. Fail fast if the file is missing.
_BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$_BRIDGE_DIR/bridge.env" ] && . "$_BRIDGE_DIR/bridge.env"

set -euo pipefail
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
REPO=${REPO:-}
for c in "$HOME/TSFM4Finance-trl/trl" "$HOME/TSFM4Finance-trl" "$HOME/tsfm-trl"; do
  [ -f "$c/scripts/sweep_evaluate.py" ] && { REPO=$c; break; }
done
[ -n "$REPO" ] || { echo "FATAL: no harness clone (looked for scripts/sweep_evaluate.py under \$HOME)"; exit 6; }
timeout 200 git -C "$REPO" pull -q --ff-only origin trl-integration 2>/dev/null \
  || echo "WARN: repo not advanced (offline or diverged); rows carry the sha printed below"

MARKER='@@'
# A row says which machine scored it, so a job that runs on the wrong box writes a wrong table.
HOST_NOW=$(hostname)
case "$HOST_NOW" in
  *gpu7*|*spark*) : ;;
  *) echo "FATAL: this job is written for the Spark worker, hostname=$HOST_NOW"; exit 6 ;;
esac
echo "-- host: $HOST_NOW"
# filled by probes/trux/bridge/gpu7_generate.sh from bridge.env on trux: this file runs on gpu7,
# where no bridge.env exists, so `:?` would only turn a generation mistake into a runtime failure
TRUX=${TRUX:-"@@TRUX@@"}
TRUX_SSH=${TRUX_SSH:-"@@TRUX_SSH@@"}
SRC=${SRC:-"@@SRC@@"}
# how *this* box reaches trux (its own key), not how trux reaches this box
GPU_SSH=${GPU_SSH:-"@@GPU_SSH@@"}
for _v in TRUX TRUX_SSH GPU_SSH SRC; do
  case "${!_v}" in
    *"$MARKER"*) echo "FATAL: $_v still holds a generator marker; run gpu7_generate.sh"; exit 6 ;;
  esac
done
DATA=${DATA:-$HOME/tsfm_data/standard}
OUTDIR=${OUTDIR:-$HOME/spark_bridge/outbox/results}
# inlined at generation time from probes/trux/gpu7_full66.series (trux's pass-3 66 series)
SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}
case " $SERIES_LIST " in
  *"$MARKER"*) echo "FATAL: SERIES_LIST still holds a generator marker; generate this job"; exit 6 ;;
esac
n_ser=$(echo $SERIES_LIST | wc -w)
echo "host=$(hostname) repo=$REPO sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "date=$(date -u +%FT%TZ)  series=$n_ser"

echo "-- corpus: $n_ser arena dirs, target $DATA (source $TRUX:$SRC)"
mkdir -p "$DATA"
missing=0
for s in $SERIES_LIST; do
  # skip what is already here: this box keeps its corpus between jobs, and the pull needs a key
  # only this box can read, so a needless pull is a needless way to fail a lane (job_015)
  [ -f "$DATA/$s/train.parquet" ] && continue
  missing=$((missing+1))
  rsync -e "$GPU_SSH" -a "$TRUX:$SRC/$s/" "$DATA/$s/" || { echo "FATAL: corpus pull failed for $s"; exit 5; }
done
echo "  pulled: $missing | present: $(ls -1 "$DATA" | wc -l)"

DEVICE=${DEVICE:-}
if [ -z "$DEVICE" ]; then
  DEVICE=$("$ENVROOT/core/bin/python" -c '
import sys, torch
if not torch.cuda.is_available():
    print("cpu"); sys.exit()
print("cuda" if (torch.cuda.mem_get_info(0)[0] >> 20) >= 4000 else "cpu")
' 2>/dev/null || echo cpu)
fi
echo "-- device: $DEVICE (torch-selected, 4 GiB floor)"

MODELS=naive,seasonal_naive,arima,theta,chronos-bolt-small,chronos-bolt-base,chronos-2,chronos-t5-small
ARGS=()
for s in $SERIES_LIST; do ARGS+=(--series "$DATA/$s"); done
for MW in 32 64; do
  OUT=$OUTDIR/gpu7_pass7_win$MW.jsonl
  echo "== windows=$MW -> $OUT"
  # each window count is its own lane file on purpose: same protocol signature means
  # mergeable rows, and 32-window rows must never supersede 8-window ones
  "$ENVROOT/core/bin/python" -u "$REPO/scripts/sweep_evaluate.py" \
    "${ARGS[@]}" --envs "$ENVROOT" --models "$MODELS" \
    --horizons 21,63 --max-windows "$MW" --ctx 256 --seed 0 \
    --out "$OUT" --device "$DEVICE" --fresh
  "$ENVROOT/core/bin/python" - "$OUT" <<'SUMMARY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
meta = next((r["meta"] for r in recs if "meta" in r), {})
scored = [r for r in recs if r.get("rows")]
cells = sum(len(r["rows"]) for r in scored)
nw = sorted({row.get("n_windows") for r in scored for row in r["rows"]})
errs = [r.get("model", "<meta line>") for r in recs if not r.get("rows")]
print(f"  records={len(recs)} scored={len(scored)} cells={cells} errors={errs}")
print(f"  lane max_windows={meta.get('max_windows')} row n_windows={nw} "
      f"machine={meta.get('machine')} sha={meta.get('code_sha')} device={meta.get('device')}")
row = scored[0]["rows"][0]
for k in ("model_key", "machine", "code_sha", "data_kind", "data_hash", "n_windows"):
    print(f"  row {k}={row.get(k)}")
SUMMARY
  cp "$OUT" "$HOME/spark_bridge/outbox/" 2>/dev/null || true
done
