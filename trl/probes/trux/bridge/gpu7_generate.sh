#!/usr/bin/env bash
# Turn a bridge job TEMPLATE into the single file that may cross to the Spark worker.
#   gpu7_generate.sh template.sh series_file out.sh
#
# Why this exists: exactly one file crosses the bridge, so every argument the job needs must be
# inside it. Two kinds of argument were a trap in sequence - the series list (job_012: a guard that
# compared against the placeholder it was supposed to detect) and the machine endpoints (job_014:
# TRUX/TRUX_SSH/SRC come from an untracked bridge.env that exists on trux and nowhere on gpu7, so
# the job died on line 40 of a script that passed `bash -n`). The template keeps the placeholders;
# this script fills them from bridge.env and refuses to emit a file that still has one.
set -euo pipefail
tpl=${1:?template required}; ser=${2:?series file required}; out=${3:?output required}
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/bridge.env" ] || { echo "FATAL: $HERE/bridge.env missing (see bridge.env.example)" >&2; exit 6; }
# shellcheck disable=SC1090
. "$HERE/bridge.env"
for v in TRUX TRUX_SSH GPU_SSH SRC; do
  [ -n "${!v:-}" ] || { echo "FATAL: bridge.env does not define $v" >&2; exit 6; }
done
series=$(cat "$ser" | tr '\n' ' ' | tr -s ' ')
python3 - "$tpl" "$series" "$out" "$TRUX" "$TRUX_SSH" "$GPU_SSH" "$SRC" <<'PY'
import pathlib, sys
tpl, series, out, trux, trux_ssh, gpu_ssh, src = sys.argv[1:8]  # argv[0] is "-"
text = pathlib.Path(tpl).read_text()
# every occurrence, deliberately: a placeholder left in a comment would otherwise survive into the
# generated file and look like the substitution half-worked
for tok, val in (("@@SERIES@@", series), ("@@TRUX@@", trux),
                 ("@@TRUX_SSH@@", trux_ssh), ("@@GPU_SSH@@", gpu_ssh), ("@@SRC@@", src)):
    text = text.replace(tok, val)
import re
# any @@UPPERCASE@@ token, not just the four known ones: a placeholder the generator has never
# heard of is exactly the kind of mistake that would otherwise be dispatched and fail remotely
left = sorted(set(re.findall(r"@@[A-Z_]+@@", text)))
if left:
    sys.exit(f"FATAL: template still contains {left}; the generator does not fill them")
pathlib.Path(out).write_text(text)
print(f"generated {out}: {len(series.split())} series, endpoints from bridge.env")
PY
bash -n "$out" && echo "syntax ok"
