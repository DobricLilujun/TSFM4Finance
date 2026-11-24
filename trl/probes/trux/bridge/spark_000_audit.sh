#!/usr/bin/env bash
# First job for the Spark: report what this machine actually is, before any row is
# trusted. The DGX lesson was that a family env can silently lose its GPU (toto,
# timesfm20 and new resolved 2.7.0+cpu / 2.8.0+cpu), so per-env torch/CUDA is probed
# rather than assumed, and the code SHA is recorded because a cross-machine table
# without it is not reproducible.
REPO=${REPO:-$HOME/tsfm-trl}
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
echo "host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
echo "date=$(date -u +%FT%TZ)"
echo "uptime: $(uptime)"
[ -d "$REPO/.git" ] && echo "code_sha=$(git -C "$REPO" rev-parse HEAD) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)" || echo "MISSING REPO at $REPO"
echo "--- GPUs:"; nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  /'
echo "--- family envs under $ENVROOT:"
for e in "$ENVROOT"/*/; do
  [ -x "$e/bin/python" ] || continue
  printf '  %-12s %s\n' "$(basename "$e")" "$("$e/bin/python" -c 'import torch;print(torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | head -1)"
done
echo "--- datasets visible here (expect 0 unless the corpus was synced):"
find "$HOME" -maxdepth 3 -type d -name standard 2>/dev/null | head -3 | sed 's/^/  /'
echo "--- disk: $(df -h "$HOME" | tail -1)"
