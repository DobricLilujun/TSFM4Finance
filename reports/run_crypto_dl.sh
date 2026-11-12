#!/usr/bin/env bash
# Crypto expansion per docs/11_data_plan.md (N1,N3,N4,N5,N6). Resumable: raw files are
# cached under data/raw and finished datasets are skipped, so re-running continues.
cd "$(dirname "$0")/.."
.venv2/bin/python -m tsfm4finance.bench.download.crypto --what pegs,funding,fut_klines,spot_daily --workers 6
.venv2/bin/python -m tsfm4finance.bench.download.crypto --what fresh --workers 4
.venv2/bin/python -m tsfm4finance.bench.download.crypto --what spot_1h --workers 6
echo CRYPTO_DL_DONE
