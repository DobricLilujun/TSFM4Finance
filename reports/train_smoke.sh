#!/usr/bin/env bash
cd /home/ubadmin/projects/TSFM4Finance
S=sp500_daily
run(){ env=$1; shift; echo "#### $env $*"; timeout 900 $env/bin/python -m tsfm4finance.bench.trainers --smoke --ignore-env --bs 2 --ctx 256 --horizon 21 --series $S "$@" 2>&1 | grep -E '^\{|step|Error|error' | tail -4; }
run .venv2     --model chronos-2 --budget fs16
run .venv-ttm  --model ttm-r2 --budget fs16
run .venv-moirai --model moirai-2.0-R-small --budget fs16
run .venv-moment --model moment-1-large --budget fs16
run .venv-toto --model toto-2-22m --budget fs16
run .venv-kronos --model kronos-small --budget fs16 --series sp500_long
S2=sp500_long
run .venv      --model timesfm-2.5-200m --budget fs16
echo TRAIN_SMOKE_DONE
