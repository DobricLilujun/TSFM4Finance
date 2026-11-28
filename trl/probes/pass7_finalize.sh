#!/usr/bin/env bash
# Pass 7 finalize: pull the 32-window lanes from both machines, cross-check them, and print the
# numbers the docs need. One command so the analysis is not re-derived by hand at the keyboard.
#
# Why a script: pass 7 changed the protocol (8 -> 32 windows), which means the *whole* pipeline -
# cross-machine check, merge, bootstrap - has to be re-run against a different lane set. Doing it by
# hand is how a 32-window row ends up superseding an 8-window one in a merged table.
#
# Usage: TSFMTRL_PY=/path/to/python probes/pass7_finalize.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 7
PY=${TSFMTRL_PY:-/home/ubadmin/projects/TSFM4Finance/.venv/bin/python}
[ -x "$PY" ] || { echo "FATAL: no harness interpreter at $PY (set TSFMTRL_PY)" >&2; exit 8; }
WATSON=${WATSON:-lujun.li@watson.uni.lux}
TRUX=${TRUX:-yewei@trux-tsubame.uni.lux}
mkdir -p reports/pass7b_watson_32 reports/pass7b_trux_32

echo "== fetch watson pass-7b lane"
timeout 900 scp -q -o BatchMode=yes "$WATSON:/home/lujun.li/pass7b_watson_32/lane32.jsonl" \
  reports/pass7b_watson_32/lane32.jsonl || echo "WARN: watson lane not ready"
echo "== fetch trux pass-7b lanes"
timeout 900 scp -q -o BatchMode=yes "$TRUX:/raid/yewei/pass7b_trux_32/lane_cuda-a.jsonl" \
  reports/pass7b_trux_32/ || echo "WARN: trux lane cuda-a not ready"
timeout 900 scp -q -o BatchMode=yes "$TRUX:/raid/yewei/pass7b_trux_32/lane_cuda-b.jsonl" \
  reports/pass7b_trux_32/ || echo "WARN: trux lane cuda-b not ready"
timeout 900 scp -q -o BatchMode=yes "$TRUX:/raid/yewei/pass7b_trux_32/lane_cpu.jsonl" \
  reports/pass7b_trux_32/ || echo "WARN: trux lane cpu not ready"

LANES=$(ls -1 reports/pass7_watson_win/lane_win32.jsonl reports/pass7_trux_win/lane_cuda-?.jsonl \
        reports/pass7b_watson_32/lane32.jsonl reports/pass7b_trux_32/lane_*.jsonl 2>/dev/null |
        tr '\n' ',' | sed 's/,$//')
[ -n "$LANES" ] || { echo "FATAL: no pass-7 lanes on disk" >&2; exit 6; }
echo "== lanes: $LANES"

echo "== cross-machine reproducibility at 32 windows"
"$PY" probes/trux/cross_machine_check.py ${LANES//,/ } --tolerance 0.02 || true

echo "== ranking at one protocol (32 windows), for the site table"
"$PY" scripts/sweep_evaluate.py --report "$LANES" --summary-only --md EFFECTIVENESS.md 2>&1 | tail -3

echo "== series-level confidence intervals at 32 windows"
"$PY" probes/pass4_share_bootstrap.py ${LANES//,/ } 2>&1 | tail -30

echo
echo "next: docs/09 'What passes 4-6 show' needs the 32-window table pasted in, and the 8-window"
echo "table either retired or kept with its window count in the header (the win column does that)."
