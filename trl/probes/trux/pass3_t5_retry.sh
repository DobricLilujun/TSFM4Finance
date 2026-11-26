#!/usr/bin/env bash
# Third attempt at the missing chronos-t5-base series, straight to CPU: two GPU attempts
# died to another user's job growing to 26 GiB on the same card. CPU rows are labelled
# device=cpu in their provenance rather than mixed invisibly into the GPU rows.
cd /raid/yewei/projects_yewei/TSFM4Finance-trl/trl
unset CUDA_VISIBLE_DEVICES
OMP_NUM_THREADS=16 /raid/yewei/envs/tsfm-trl/core/bin/python -u scripts/sweep_evaluate.py \
  $(sed 's/^/--series /' /tmp/pass3_t5_missing.txt) --models chronos-t5-base \
  --horizons 21,63 --max-windows 8 --batch 4 --seed 0 --device cpu --fresh \
  --out reports/pass3_trux_t5retry/lane_t5retry.jsonl
echo "exit=$? rows=$(wc -l < reports/pass3_trux_t5retry/lane_t5retry.jsonl)"
