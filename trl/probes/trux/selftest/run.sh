#!/usr/bin/env bash
# Self-test for trux_lane_runner.py, runnable on any of the machines (no weights, no GPU).
# It checks the parts that actually break: the flag contract with sweep_evaluate.py,
# lane splitting under a memory floor, resume-skipping, and the contention sidecar.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE=$(dirname "$HERE")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/repo/scripts" "$TMP/envs/core/bin" "$TMP/out"
cp "$HERE/stub_sweep.py" "$TMP/repo/scripts/sweep_evaluate.py"
ln -s "$(command -v python3)" "$TMP/envs/core/bin/python"
head -3 "$PROBE/sweep_stratified.txt" > "$TMP/three.txt"
R="python3 $PROBE/trux_lane_runner.py --repo $TMP/repo --env-root $TMP/envs --min-free-mib 99999 --lanes cpu,cpu2 --out-dir $TMP/out"

echo "1. plan + run:"
$R --series-file "$TMP/three.txt" 2>&1 | grep -E "lane|total wall" | sed 's/^/   /'
rows=$(cat $(ls "$TMP"/out/lane_*.jsonl 2>/dev/null | grep -v '\.env\.jsonl$') 2>/dev/null | wc -l)
[ "$rows" -ge 9 ] && echo "   OK rows=$rows (3 series x 3 horizons, sidecars excluded)" || { echo "   FAIL rows=$rows"; exit 1; }

echo "2. resume must not re-score:"
$R --series-file "$TMP/three.txt" 2>&1 | grep -E "resume" | sed 's/^/   /'
rows2=$(cat $(ls "$TMP"/out/lane_*.jsonl 2>/dev/null | grep -v '\.env\.jsonl$') 2>/dev/null | wc -l)
[ "$rows2" = "$rows" ] && echo "   OK still $rows2 rows" || { echo "   FAIL grew to $rows2"; exit 1; }

echo "3. negative test - a flag neither the stub nor the real CLI has must fail:"
# its own out-dir: with the previous lanes' JSONL present, resume would skip every
# series and the stub would never run, which looks identical to "the flag was accepted"
python3 "$PROBE/trux_lane_runner.py" --repo "$TMP/repo" --env-root "$TMP/envs" \
    --min-free-mib 99999 --lanes cpu --out-dir "$TMP/negout" \
    --series-file "$TMP/three.txt" --sweep-arg=--no-such-flag=1 > "$TMP/neg.log" 2>&1
# the child's stderr goes to the lane log, not to the runner's stdout — check the lane log
if grep -qi "unrecognized arguments" "$TMP/negout/lane_cpu.log" 2>/dev/null \
   && grep -qE "lane cpu: rc=[1-9]" "$TMP/neg.log"; then
  echo "   OK rejected: $(grep -o 'unrecognized arguments.*' "$TMP/negout/lane_cpu.log" | head -1)"
  grep -oE "lane cpu: rc=[0-9]+" "$TMP/neg.log" | sed 's/^/      /'
else
  echo "   FAIL expected an argparse rejection in the lane log"
  tail -3 "$TMP/neg.log" 2>/dev/null; tail -3 "$TMP/negout/lane_cpu.log" 2>/dev/null; exit 1
fi

echo "   note: the real CLI inherits argparse prefix matching, so --windows=3 would be"
echo "         silently read as --max-windows=3 there; the stub forbids abbreviations"

echo "4. contention sidecar (a timing number needs its context):"
python3 - "$TMP/out/lane_cpu.env.jsonl" <<'PY'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
last = lines[-1]
need = {"free", "other_processes", "loadavg", "rc"}
missing = need - set(last)
print("   free:", [g["free_mib"] for g in last["free"]], "loadavg:", last["loadavg"])
print("   OK sidecar complete" if not missing else f"   FAIL missing {missing}")
sys.exit(0 if not missing else 1)
PY

echo "5. dry-run needs no weights and no real sweep script:"
python3 "$PROBE/trux_lane_runner.py" --series-file "$TMP/three.txt" --dry-run \
    --out-dir "$TMP/dry" 2>&1 | grep -E "lane |dry run" | sed 's/^/   /'
echo "self-test passed"
