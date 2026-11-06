#!/usr/bin/env bash
set -u
cd /home/ubadmin/projects/TSFM4Finance
run() { echo "#### $1 : $2"; $1/bin/python -m tsfm4finance.bench.doctor --weights --families "$2" --python "$1" 2>&1 | tail -20; }
run .venv2 "baseline,timesfm25,chronos,chronos_bolt,timemoe"
run .venv2 "timesfm30"
run .venv-moirai "moirai"
run .venv-moment "moment"
run .venv-ttm "ttm,flowstate"
run .venv-toto "toto2"
run .venv-tirex "tirex"
run .venv-tirex2 "tirex2"
run .venv-kronos "kronos"
echo "ALL_DONE"
