#!/usr/bin/env bash
# Inference-only pass over tiers A,B x core12+long12 x h in {5,21}, zero-shot.
# Each stage is resumable: re-running after an interrupt continues where it stopped.
cd /home/ubadmin/projects/TSFM4Finance
# local model weights only - nothing is downloaded during the run
# (see docs/08_offline_models.md; drop these two lines if you DO want HF downloads)
export TSFM_WEIGHTS_ROOT=$PWD/models TSFM_OFFLINE=1
export HF_HOME=${HF_HOME:-$PWD/.hf-cache}
KEEP=$(cat reports/models_pass1.txt)
for STAGE in forecast trading classify anomaly; do
  echo "==== $STAGE $(date -Is)"
  .venv2/bin/python -m tsfm4finance.bench.sweep --stage $STAGE --workers 1 \
      --models "$KEEP" --ctx-cap 512 --batch 16 --device cuda --amp fp32
  .venv2/bin/python -m tsfm4finance.bench.sweep --stage status >/dev/null
done
echo SWEEP_PASS1_DONE
