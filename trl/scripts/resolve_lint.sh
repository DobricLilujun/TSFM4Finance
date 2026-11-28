#!/usr/bin/env bash
# Print the ruff invocation this machine can actually run, on stdout. Non-zero means there is
# none, and the caller must fail - not skip.
#
# Lint used to be the first gate in scripts/check.sh and the first thing to quietly disappear:
# on trux the harness env has no ruff, so `$PY -m ruff` died, the gate stopped at lint, and
# "all gates green" had silently stopped meaning linted. A missing linter has to be an error with
# an install command in it, never a pass.
set -uo pipefail
PY=${1:-}
if [ -n "$PY" ] && "$PY" -m ruff --version >/dev/null 2>&1; then
  echo "$PY -m ruff"
elif command -v ruff >/dev/null 2>&1; then
  echo "ruff"
elif command -v uvx >/dev/null 2>&1; then
  # pinned: a different ruff can call the same code clean on one machine and dirty on another
  echo "uvx ruff@0.14"
else
  echo "no linter found: install one with 'uv tool install ruff@0.14' (or put ruff in \$PY)" >&2
  exit 1
fi
