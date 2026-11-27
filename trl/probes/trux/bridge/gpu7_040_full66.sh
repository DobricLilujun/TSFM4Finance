#!/usr/bin/env bash
# gpu7 (DGX Spark, GB10) over trux's full 66-series corpus, at pass-3's protocol.
#
# Why: pass-3's trux block (66 series) and watson block (35 series) were two disjoint
# partitions, and the 12-series replication proved cross-machine agreement only inside that
# overlap. This lane puts a second machine - aarch64 GB10, torch 2.9.1+cu130 - across the
# *whole* trux stratum, so any trux number can be checked against a second machine on the
# same data path (arena parquet, publisher's split) rather than assumed.
#
# Model set = the 9 already proven on this box (job_011) + two cheap core-env baselines
# (moving_average, theta). chronos-t5-base is deliberately absent: it OOMed on trux's 48 GB
# card, and gpu7's free memory swings between 1 and 8 GiB minute-to-minute. ttm-r3 absent
# for its own documented reason (non-finite windows on this corpus, docs/08).
#
# GENERATED before dispatch - one file crosses the bridge:
#
#   probes/trux/bridge/gpu7_generate.sh \
#       probes/trux/bridge/gpu7_040_full66.sh probes/trux/gpu7_full66.series /tmp/job.sh
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
# trux is the control plane and the inbox is shared state: refuse to score anywhere else.
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
OUT=${OUT:-$HOME/spark_bridge/outbox/results/gpu7_pass4_full.jsonl}
# inlined at generation time from probes/trux/gpu7_full66.series (trux's pass-3 66 series)
SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}
# The guard must not contain the placeholder itself: the generator substitutes EVERY
# occurrence, so a guard that compares against "@@SERIES@@" becomes `[ list != list ]`
# and fails a perfectly good job (job_012 died exactly this way). Test for a leftover
# marker instead - and note the marker here is spelled with the sigil split from the
# word so this file cannot self-trigger it.
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
  # skip what is already here: this box keeps its copy of the corpus between jobs, and the pull
  # path below needs credentials that only exist on this box, so a needless pull is a needless way
  # to fail a two-hour lane
  [ -f "$DATA/$s/train.parquet" ] && continue
  missing=$((missing+1))
  # GPU_SSH (not TRUX_SSH): the key baked into TRUX_SSH lives on trux and is not readable from
  # here. job_015 died exactly that way - "Identity file /raid/yewei/.ssh/... not accessible",
  # then Permission denied - over a corpus this box already had in full.
  rsync -e "$GPU_SSH" -a "$TRUX:$SRC/$s/" "$DATA/$s/" || { echo "FATAL: corpus pull failed for $s"; exit 5; }
done
echo "  pulled: $missing | present: $(ls -1 "$DATA" | wc -l)"
ls -1 "$DATA" | wc -l | sed 's/^/  series dirs present: /'

# GB10 answers [N/A] to nvidia-smi memory queries and the box is shared, so the device comes
# from torch. Free at launch is not free during the run: a CUDA row can still die mid-lane.
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

ARGS=()
for s in $SERIES_LIST; do ARGS+=(--series "$DATA/$s"); done
"$ENVROOT/core/bin/python" -u "$REPO/scripts/sweep_evaluate.py" \
  "${ARGS[@]}" \
  --envs "$ENVROOT" \
  --models naive,seasonal_naive,arima,moving_average,theta,chronos-bolt-small,chronos-bolt-base,chronos-t5-small,chronos-2,timesfm-2.5-200m,ttm-r2 \
  --horizons 21,63 --max-windows 8 --ctx 256 --seed 0 \
  --out "$OUT" --device "$DEVICE" --fresh

echo "rows=$(wc -l < "$OUT")"
"$ENVROOT/core/bin/python" - "$OUT" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
scored = [r for r in recs if r.get("rows")]
print(f"  records={len(recs)} scored={len(scored)} "
      f"cells={sum(len(r['rows']) for r in scored)} "
      f"errors={[r.get('model', '<meta line>') for r in recs if not r.get('rows')]}")
if scored:
    r = scored[0]; row = r["rows"][0]
    for k in ("model", "model_key", "machine", "device", "code_sha", "torch", "data_kind", "data_hash"):
        v = row.get(k, r.get(k))
        if v is not None:
            print(f"  {k}={v}")
PY
cp "$OUT" "$HOME/spark_bridge/outbox/" 2>/dev/null || true
