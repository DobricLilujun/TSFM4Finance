#!/usr/bin/env bash
# gpu7_030 with the series list baked in: trux_dispatch copies the job file to gpu7 and
# runs it in a fresh shell, so an env var exported on trux never arrives. The list lives in
# gpu7_watson12.series (12 of watson's 35 pass-3 series, every 3rd) next to this script.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SERIES_LIST=$(cat "$here/gpu7_watson12.series")
export SERIES_LIST
echo "series=$(echo "$SERIES_LIST" | wc -w)"
exec bash "$here/gpu7_030_replicate.sh"
