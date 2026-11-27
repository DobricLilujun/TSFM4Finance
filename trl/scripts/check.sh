#!/usr/bin/env bash
# Everything that must be green before a matrix run is trusted.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-envs/core/bin/python}
[ -x "$PY" ] || PY=${VIRTUAL_ENV:-.venv2}/bin/python

echo "== lint"
$PY -m ruff check tsfm_trl tests examples scripts probes --select F,E9 --ignore F841 || exit 1
echo "== tests"
$PY -m pytest -q tests || exit 1
echo "== registry consistency (46 rows, adapters importable in this env)"
$PY - <<'PY'
from tsfm_trl.model import ModelHub
from tsfm_trl.model.base import CAPS
reg = ModelHub.registry()
bad = []
for s in reg.models():
    if not set(s.caps) <= CAPS:
        bad.append(f"{s.key}: unknown caps {set(s.caps) - CAPS}")
    try:
        ModelHub.registry().adapter_class(s)
    except Exception as e:
        bad.append(f"{s.key}: adapter {s.adapter} -> {type(e).__name__}: {e}")
print(f"{len(reg.models())} rows, {len(bad)} broken")
for b in bad:
    print("  ", b)
raise SystemExit(1 if bad else 0)
PY
echo "== mini matrix (cpu, fixtures; the point is that every step returns a verdict)"
TSFM_WEIGHTS_ROOT=${TSFM_WEIGHTS_ROOT:-$HOME/models} $PY -m tsfm_trl.cli doctor \
    --model naive,theta,chronos-bolt-small --tasks load,predict,classify,anomaly \
    --device cpu --ctx 128 --horizon 8 --json reports/check_matrix.json --md - \
  || exit 1
echo "OK"

# Docs links/nav, if the docs toolchain is present. Not required for the library: the
# site is a view, and CI on GitHub builds it anyway (docs.yml runs --strict).
if [ -x .venv-docs/bin/python ]; then
  DOCS_PY=.venv-docs/bin/python ./scripts/build_docs.sh >/dev/null && echo "docs: OK"
fi
