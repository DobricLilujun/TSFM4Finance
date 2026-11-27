#!/usr/bin/env bash
# First job for the Spark: report what this machine actually is, before any row is
# trusted. The DGX lesson was that a family env can silently lose its GPU (toto,
# timesfm20 and new resolved 2.7.0+cpu / 2.8.0+cpu), so per-env torch/CUDA is probed
# rather than assumed, and the code SHA is recorded because a cross-machine table
# without it is not reproducible.
REPO=${REPO:-$HOME/tsfm-trl}
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
# The clone is either the standalone harness or vendored as trl/ inside TSFM4Finance, and
# the directory name differs per machine. Guessing one path produced an empty code_sha in
# the first gpu7 audit, which reads as "clean tree, unknown commit" — worse than an error.
if [ ! -d "$REPO/.git" ]; then
  for c in "$HOME/TSFM4Finance-trl" "$HOME/trl" "$HOME/projects/TSFM4Finance-trl" "$HOME/tsfm-trl"; do
    if [ -f "$c/scripts/sweep_evaluate.py" ]; then REPO="$c"; break; fi
    if [ -f "$c/trl/scripts/sweep_evaluate.py" ]; then REPO="$c/trl"; break; fi
  done
fi
echo "host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
echo "date=$(date -u +%FT%TZ)"
echo "uptime: $(uptime)"
if [ -d "$REPO/.git" ]; then
  echo "repo=$REPO code_sha=$(git -C "$REPO" rev-parse HEAD) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
else
  echo "MISSING REPO (looked at $REPO and the usual vendored paths)"
fi
echo "--- GPUs (nvidia-smi):"; nvidia-smi --query-gpu=index,name,driver_version,memory.free,memory.total,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  /'
# Unified-memory cards (DGX Spark / GB10) answer [N/A] for memory.free and memory.total
# while CUDA works normally. Ask torch for what the scheduler actually needs.
if [ -x "$ENVROOT/core/bin/python" ]; then
  echo "--- GPUs (torch, core env):"
  "$ENVROOT/core/bin/python" -c '
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  {i} {torch.cuda.get_device_name(i)} free={free>>20}MiB total={total>>20}MiB")
' 2>&1 | sed 's/^/  /'
fi
echo "--- family envs under $ENVROOT:"
for e in "$ENVROOT"/*/; do
  [ -x "$e/bin/python" ] || continue
  printf '  %-12s %s\n' "$(basename "$e")" "$("$e/bin/python" -c 'import torch;print(torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | head -1)"
done
echo "--- datasets visible here (expect 0 unless the corpus was synced):"
find "$HOME" -maxdepth 3 -type d -name standard 2>/dev/null | head -3 | sed 's/^/  /'
echo "--- disk: $(df -h "$HOME" | tail -1)"
