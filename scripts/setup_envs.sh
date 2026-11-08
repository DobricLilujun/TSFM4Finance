#!/usr/bin/env bash
# Build one uv project environment per model family (envs/<family>/.venv).
# The families cannot share an interpreter (uni2ts pins torch<2.5, momentfm pins
# numpy==1.25, toto-2 needs python>=3.12, Time-MoE needs transformers==4.44,
# TiRex-2 needs triton<3.6), so each gets its own - with pins frozen in
# envs/<family>/pyproject.toml by scripts/gen_env_projects.py.
set -uo pipefail
cd "$(dirname "$0")/.."

case "${1:-build}" in
  build)
    for p in envs/*/; do
      fam=$(basename "$p")
      echo "== uv sync --project $p"
      uv sync --project "$p" || echo "!! failed: $fam"
    done
    echo; echo "== resolved interpreters (bench/config.yaml)"
    "${CORE_PY:-.venv2/bin/python}" - <<'PY'
import os
from tsfm4finance.bench.config import load_grid
g = load_grid(); ms = g.models()
for e in sorted({m.env for m in ms.values()}):
    py = g.python_for(next(m for m in ms.values() if m.env == e))
    print(f"  {e:<9} {py:<48} exists={os.path.exists(py)}")
PY
    ;;
  check)
    .venv2/bin/python -m tsfm4finance.bench.weights --verify || true
    ./reports/doctor_all.sh
    ;;
  freeze)   # re-freeze pins from the venvs that currently work
    python3 scripts/gen_env_projects.py
    ;;
  *) echo "usage: $0 [build|check|freeze]"; exit 2;;
esac
