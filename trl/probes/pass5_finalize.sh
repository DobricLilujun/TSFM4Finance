#!/usr/bin/env bash
# After watson SLURM 3291 finishes: pull the lane, widen the cross-machine stratum, republish.
# Run from the vendored trl/ tree on the laptop. One step, no hand-editing between steps.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# report + checker need the harness's own interpreter, not the login shell's python3
PY="${TSFMTRL_PY:-/home/ubadmin/projects/TSFM4Finance/.venv/bin/python}"
LANE="$HERE/reports/pass5_watson_66/lane.jsonl"

# 1. lane from watson (git, never /tmp)
if [ ! -s "$LANE" ]; then
  scp -q -o BatchMode=yes lujun.li@watson.uni.lux:~/pass5_watson_66/lane.jsonl "$LANE"
fi
wc -l "$LANE"

# 2. the measurement pass-5 exists for: gpu7 (GB10 CUDA) vs watson (A6000 CUDA) over the SAME 66 series
"$PY" probes/trux/cross_machine_check.py \
  reports/pass4_gpu7_full/lane_full66.jsonl \
  "$LANE" --tolerance 0.02 | tail -20

# 3. merged site page: pass-5 last, so its rows win for cells gpu7 also scored
FILES="$(grep -v '^#' probes/pass4_merge_files.txt | grep -v '^$' | tr '\n' ',' | sed 's/,$//')"
"$PY" scripts/sweep_evaluate.py --report "$FILES" --summary-only --md EFFECTIVENESS.md
