#!/usr/bin/env bash
# Inference-only pass over tiers A,B x core12+long12 x h in {5,21}, zero-shot.
# Each stage is resumable: re-running after an interrupt continues where it stopped.
cd /home/ubadmin/projects/TSFM4Finance
KEEP=$(cat reports/models_pass1.txt)
for STAGE in forecast trading classify anomaly; do
  echo "==== $STAGE $(date -Is)"
  .venv2/bin/python -m tsfm4finance.bench.sweep --stage $STAGE --workers 1 \
      --models "$KEEP" --ctx-cap 512 --batch 16 --device cuda --amp fp32
  .venv2/bin/python -m tsfm4finance.bench.sweep --stage status >/dev/null
done
echo SWEEP_PASS1_DONE
