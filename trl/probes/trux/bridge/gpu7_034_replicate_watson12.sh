#!/usr/bin/env bash
# gpu7 (DGX Spark, GB10) re-scores 12 series that watson already scored, under pass-3's
# exact protocol, to turn "the machines agree" from a 3-series spot check into a stratum.
#
# Why these 12: they are watson's (sedan-gpu3, RTX A6000, torch 2.7.1+cu128) and gpu7 is
# aarch64 GB10 with torch 2.9.1+cu130 - different vendor build, different architecture,
# different node. Any gap that survives that is a hardware/build effect, and a 0.000% gap
# is what a benchmark may then claim across machines.
#
# The corpus arrives as arena parquet directories (train/validation/test + meta.json), the
# same layout the trux and watson lanes read. The earlier gpu7 smoke pulled CSVs, which is
# a second data path: fine for a smoke, not for a comparison cell.
# ttm-r3 is deliberately absent: it returns non-finite windows on this corpus (docs/08),
# and its cells would change the scored-window denominator rather than test agreement.
set -euo pipefail
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
REPO=${REPO:-}
for c in "$HOME/TSFM4Finance-trl/trl" "$HOME/TSFM4Finance-trl" "$HOME/tsfm-trl"; do
  [ -f "$c/scripts/sweep_evaluate.py" ] && { REPO=$c; break; }
done
[ -n "$REPO" ] || { echo "FATAL: no harness clone (looked for scripts/sweep_evaluate.py under \$HOME)"; exit 6; }
# The row records this clone's sha, so advance it before running: a lane labelled with a
# stale commit is not reproducible even when its numbers are right. Best-effort - an
# offline Spark should still produce rows, tagged with whatever sha it does have.
timeout 200 git -C "$REPO" pull -q --ff-only origin trl-integration 2>/dev/null \
  || echo "WARN: repo not advanced (offline or diverged); rows carry the sha printed below"

TRUX=${TRUX:-yewei@trux-tsubame.uni.lux}
TRUX_SSH=${TRUX_SSH:-"ssh -i $HOME/.ssh/id_ed25519_gm_transfer -o StrictHostKeyChecking=accept-new"}
SRC=${SRC:-/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard}
DATA=${DATA:-$HOME/tsfm_data/standard}
OUT=${OUT:-$HOME/spark_bridge/outbox/results/gpu7_replicate.jsonl}
# GENERATED from gpu7_030_replicate.sh + gpu7_watson12.series (12 watson pass-3 series).
# Inline: trux_dispatch transfers exactly one file, so nothing around the job file
# crosses the bridge (see docs/08, jobs 006-008). Edit the .series file and regenerate.
SERIES_LIST=${SERIES_LIST:-"std_equity_1d_10104 std_equity_1d_14008 std_equity_1d_20482 std_equity_1d_47896 std_equity_1d_66157 std_equity_1d_89525 std_fi_1d_uk_5y std_fi_1d_us_2y std_fi_1d_jp_7y std_fx_1d_eurusd std_fx_1d_usdmxn std_fx_1h_usdchf"}
[ -n "$SERIES_LIST" ] || { echo "FATAL: SERIES_LIST (space-separated series dir names) required"; exit 6; }

echo "host=$(hostname) repo=$REPO sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "date=$(date -u +%FT%TZ)"

echo "-- corpus: ${SERIES_LIST} from $TRUX:$SRC"
mkdir -p "$DATA"
for s in $SERIES_LIST; do
  rsync -e "$TRUX_SSH" -a "$TRUX:$SRC/$s" "$DATA/" || { echo "FATAL: corpus pull failed for $s"; exit 5; }
done
ls -1 "$DATA" | wc -l | sed 's/^/  series dirs present: /'

# GB10 answers [N/A] to nvidia-smi memory queries and the box is shared, so the device
# comes from torch, and the choice is printed before the run rather than inferred later.
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
  --models naive,seasonal_naive,arima,chronos-bolt-small,chronos-bolt-base,chronos-t5-small,chronos-2,timesfm-2.5-200m,ttm-r2 \
  --horizons 21,63 --max-windows 8 --ctx 256 --seed 0 \
  --out "$OUT" --device "$DEVICE" --fresh

echo "rows=$(wc -l < "$OUT")"
head -1 "$OUT" | tr ',' '\n' | grep -iE '"device|"torch|"machine|"code_sha|"interpreter' | head -6 | sed 's/^/  /'
cp "$OUT" "$HOME/spark_bridge/outbox/" 2>/dev/null || true
