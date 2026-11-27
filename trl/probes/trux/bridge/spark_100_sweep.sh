#!/usr/bin/env bash
# The Spark's half of pass 2: a disjoint series set, so trux + Spark form a union
# instead of duplicating work. Results stay in JSONL; the harness merges per
# (model, series, span, horizon) and groups provenance by (interpreter, machine, device).
set -euo pipefail
REPO=${REPO:-}
# The harness is either the repo root (standalone tsfm-trl) or the vendored subtree
# (TSFM4Finance/trl), and the clone directory name differs per machine. Scan for the
# entry point instead of hardcoding one path, so a correct clone never reads as
# "code not pulled yet".
for c in "$HOME/TSFM4Finance-trl/trl" "$HOME/TSFM4Finance-trl" "$HOME/tsfm-trl" "$HOME/projects/TSFM4Finance-trl/trl"; do
  [ -f "$c/scripts/sweep_evaluate.py" ] && { REPO=$c; break; }
done
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
# the series list is embedded, so the job is one self-contained file: a worker that
# cannot fetch extra inputs still runs exactly the intended subset
SERIES=${SERIES:-$HOME/spark_bridge/pass2_spark.txt}
mkdir -p "$(dirname "$SERIES")"
cat > "$SERIES" <<'SERIES_EOF'
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard/std_equity_1d_10107
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard/std_equity_1d_11850
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard/std_fi_1d_uk_20y
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard/std_fx_1d_euraud
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/standard/std_fx_1h_eurchf
/raid/yewei/projects_yewei/TSFM4Finance/assets/data/open/btc_crypto
SERIES_EOF
OUT=${OUT:-$HOME/spark_bridge/outbox/results/pass2_spark.jsonl}
[ -f "$SERIES" ] || { echo "FATAL: $SERIES not on the Spark yet (push it with the job)"; exit 5; }
[ -f "$REPO/scripts/sweep_evaluate.py" ] || { echo "FATAL: no $REPO — code not pulled yet"; exit 6; }
mkdir -p "$(dirname "$OUT")"
# a CPU-only torch on a GPU-flagged run is how bad rows got published before: stop here
if ! "$ENVROOT/core/bin/python" -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 9)'; then
  echo "FATAL: $ENVROOT/core has no CUDA; refusing to write silently-CPU rows"
  exit 9
fi
"$ENVROOT/core/bin/python" "$REPO/scripts/sweep_evaluate.py" \
  --horizons 21,63,126 --windows 12 --seed 0 --device cuda --out "$OUT" \
  $(cat "$SERIES")
echo "rows=$(wc -l < "$OUT")"
# the JSONL travels back through the worker's outbox rsync
cp "$OUT" "$(dirname "$OUT")/../" 2>/dev/null || true
