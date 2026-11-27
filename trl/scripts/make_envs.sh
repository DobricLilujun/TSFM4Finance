#!/usr/bin/env bash
# Create the per-family venvs declared in envs/envs.toml.
#   ./scripts/make_envs.sh                 # every env, into ./envs/<name>
#   ./scripts/make_envs.sh core moirai     # a subset
#   TORCH_SPEC="torch==2.9.1" TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#       ./scripts/make_envs.sh core        # aarch64 / Blackwell (DGX Spark)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
TOOL=${UV:-uv}
TORCH_SPEC=${TORCH_SPEC:-torch==2.8.0}
TORCH_INDEX=${TORCH_INDEX:-}
DRY=${DRY:-}

pick() { # $1 = env name -> prints the toml block
  python3 - "$1" <<'PY'
import sys, tomllib
spec = tomllib.load(open("envs/envs.toml", "rb"))
name = sys.argv[1]
env = spec["env"][name]
print(env.get("python", "3.11"))
print("\n".join(env.get("deps", [])))
PY
}

names=${*:-$($TOOL --version >/dev/null && python3 -c "
import tomllib;print(' '.join(tomllib.load(open('envs/envs.toml','rb'))['env']))")}

for name in $names; do
  venv="envs/$name"
  py=$(pick "$name" | head -1)
  echo "== $name (python $py) -> $venv"
  [ -n "$DRY" ] && { pick "$name" | tail -n +2 | sed 's/^/   /'; continue; }
  $TOOL venv --python "$py" "$venv"
  # torch first, from the CUDA index if one was given, so the resolver never replaces it
  if [ -n "$TORCH_INDEX" ]; then
    $TOOL pip -q install --python "$venv/bin/python" "$TORCH_SPEC" --index-url "$TORCH_INDEX"
  fi
  spec_install=$(python3 - "$name" <<'PYSPEC'
import sys, tomllib
spec = tomllib.load(open("envs/envs.toml", "rb"))["env"][sys.argv[1]]
if spec.get("lock"):
    # --no-deps: a freeze is a *snapshot* of a venv where the matrix passed, and its
    # metadata can be mutually contradictory (gluonts asks pandas<2.2 while the working
    # venv carries 2.3.3). Re-resolving would refuse it; installing it verbatim works.
    print(f"--no-deps -r envs/{spec['lock']}")
else:
    print(" ".join(f'"{d}"' for d in spec.get("deps", []) if d != "torch"))
PYSPEC
)
  [ -n "$spec_install" ] && eval "$TOOL pip -q install --python '$venv/bin/python' $spec_install"
  [ "$TORCH_INDEX" ] || $TOOL pip -q install --python "$venv/bin/python" "$TORCH_SPEC"
  # the library itself, editable, so the matrix can import it from every env
  $TOOL pip -q install --python "$venv/bin/python" -e .
  # git-only families
  python3 - "$name" "$ROOT/third_party" <<'PY'
import os, subprocess, sys, tomllib
spec = tomllib.load(open("envs/envs.toml", "rb"))["env"].get(sys.argv[1], {})
dest = sys.argv[2]
for g in spec.get("git", []):
    d = os.path.join(dest, g["dir"])
    if not os.path.isdir(d):
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", g["repo"], d], check=False)
PY
  # env vars declared for this family (e.g. USE_TRITON_KERNELS)
  python3 - "$name" "$venv" <<'PY'
import json, sys, tomllib
spec = tomllib.load(open("envs/envs.toml", "rb"))["env"].get(sys.argv[1], {})
if spec.get("env"):
    open(f"{sys.argv[2]}/lib/tsfm_env.json", "w").write(json.dumps(spec["env"]))
PY
done
echo
echo "done. Then:"
echo "  export TSFM_WEIGHTS_ROOT=\$HOME/models   TSFMTRL_THIRD_PARTY=\$PWD/third_party"
echo "  python -m tsfm_trl.cli doctor --all-envs --md COMPATIBILITY.md"
