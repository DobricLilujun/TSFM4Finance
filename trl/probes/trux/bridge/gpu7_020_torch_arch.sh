#!/usr/bin/env bash
# Does any installable torch actually run on GB10 (cuda capability 12.1)?
#
# The pinned core lock ships torch 2.9.1+cu130, whose supported range is (8.0)-(12.0).
# On the Spark that means: torch.cuda.is_available() is true, get_device_name() prints,
# and the first real CUDA call (mem_get_info) dies with "out of memory" — an unsupported
# architecture reported as an allocation failure. A benchmark row from that env would be
# either CPU-labelled or wrong, so find the build that covers sm_121 before running a
# sweep here.
set -uo pipefail
ENVROOT=${ENVROOT:-$HOME/envs/tsfm-trl}
PY=$ENVROOT/core/bin/python
echo "host=$(hostname) date=$(date -u +%FT%TZ)"
echo "-- driver:"; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 >/dev/null
echo "-- current core env:"
"$PY" -c 'import torch;print("  torch",torch.__version__);print("  arch_list",torch.cuda.get_arch_list());print("  is_available",torch.cuda.is_available())' 2>&1 | sed 's/^/  /'
echo "-- capability and first real CUDA call:"
"$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import torch
try:
    print("capability", torch.cuda.get_device_capability(0))
    x = torch.randn(64, 64, device="cuda")
    print("matmul ok", float((x @ x).sum()))
    print("mem", [v >> 20 for v in torch.cuda.mem_get_info(0)], "MiB")
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
PY
echo "-- what the cu130 index offers for aarch64:"
timeout 300 "$PY" -m pip index versions torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | head -3 | sed 's/^/  /'
echo "-- build a scratch env with the newest available torch and re-probe:"
SCRATCH=$HOME/envs/tsfm-trl/_arch_probe
rm -rf "$SCRATCH"
timeout 900 "$PY" -m venv "$SCRATCH" >/dev/null 2>&1
timeout 1500 "$SCRATCH/bin/python" -m pip install -q --index-url https://download.pytorch.org/whl/cu130 torch 2>&1 | tail -2 | sed 's/^/  /'
"$SCRATCH/bin/python" - <<'PY' 2>&1 | sed 's/^/  /'
import torch
print("torch", torch.__version__, "arch_list", torch.cuda.get_arch_list())
try:
    x = torch.randn(256, 256, device="cuda")
    print("matmul ok", float((x @ x).sum()), "mem_mib", [v >> 20 for v in torch.cuda.mem_get_info(0)])
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
PY
rm -rf "$SCRATCH"
