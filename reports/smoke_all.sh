#!/usr/bin/env bash
cd /home/ubadmin/projects/TSFM4Finance
run(){ echo "#### $1"; $1/bin/python -m tsfm4finance.bench.smoke --models "$2" --series sp500_long --horizon 21 --n 4 2>&1 | grep -E '^[a-z].*\{' | cut -c1-600; }
run .venv-ttm  "ttm-r2,flowstate-r1"
run .venv-toto "toto-2-22m"
run .venv-moirai "moirai-2.0-R-small"
run .venv-moment "moment-1-large"
run .venv-tirex "tirex"
run .venv-tirex2 "tirex-2"
run .venv-timemoe "timemoe-50m"
run .venv-kronos "kronos-small"
echo SMOKE_DONE
