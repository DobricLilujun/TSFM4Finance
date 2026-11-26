# Pass 3: the full `standard/` corpus (101 series), two machines

Protocol identical to pass 2 (`docs/09_benchmark.md`): zero-shot, `--spans all`,
horizons 21 and 63, 8 windows per series, ctx 256, batch 16 (4 in the GPU retry), seed 0.
One row per (model, series, span, horizon); multi-machine results merge per cell.

## Partition

`pass3_trux.txt` 66 series, `pass3_watson.txt` 35 series, **disjoint**, allocated 2:1
within every stratum (equity 33/17, sovereign-yield 14/8, FX daily 13/7, FX hourly 6/3) so
that trux's two GPU lanes and watson's one A6000 lane both cover every dataset family.
Allocation rule is length-descending and independent of any model result.

## Model coverage differs by machine, and that is recorded rather than smoothed over

| machine | envs present | models |
|---|---|---|
| trux | `core`, `moment`, `timemoe` | 19 (incl. MOMENT-1 small/large, TimeMoE-50M) |
| watson | `core` only | 16 |

A model absent from a machine's table is an environment fact, not a verdict. The report's
"Not in this table" section and the per-interpreter provenance table are what say so.

## Files

| file | what |
|---|---|
| `lane_cuda-a.jsonl`, `lane_cuda-b.jsonl` | trux GPU lanes, code 1d2f3ae |
| `../pass3_trux_moment/lane_cuda-*.jsonl` | MOMENT re-run after the `_pipe_key` fix, code 41f3340. **Scored on CPU**: both cards were held by other users at ~1.1 GiB free, and the runner's `--min-free-mib` guard downgraded the lanes with a warning instead of launching an OOM |
| `../pass3_trux_t5retry/lane_t5retry.jsonl` | `chronos-t5-base` retry for the 33 series that hit CUDA OOM (below) |
| `../pass3_watson_src/csv/` | the 35 watson CSVs, generated on trux and relayed through the laptop (trux has no route to watson) |
| `csv/`, `csv/MANIFEST.json` | flattened series + per-series split diagnostics |

## What the shared cards did to the run

- trux `chronos-t5-base`: 33 of 66 series failed with `torch.OutOfMemoryError`. The error
  says the card had **44.94 MiB free at that moment** while `nvidia-smi` minutes earlier
  reported 10 GiB — another user's process grew from 18.65 GiB to 26 GiB during the
  attempt, and our own process peaked at 7.09 GiB. Lesson: on a shared card, free memory
  *at launch* is not free memory *during the run*; the guard is necessary but not
  sufficient, and a mid-run OOM is an environment event to be re-run, never a model result.
- The first retry picked "cuda" from the total and landed on the contended GPU 0 — a GPU
  lane needs `CUDA_VISIBLE_DEVICES` pinned to the chosen index, not just a free-memory
  check. Second retry pins it and lowers `--batch`.
- MOMENT's 132 pairs therefore come from CPU lanes and carry `device: cpu`; they are
  comparable to each other, and the provenance table separates them from the GPU rows.

## Verification that the new labels work

264 of 264 MOMENT rows carry `meta: {"head_from_checkpoint": false,
"head_trained_here": false}`, and the merged report lists them under "Flagged: the scored
head was never trained" while keeping the numbers. Before that fix landed, the same rows
were green cells scoring a randomly initialised head — which is exactly the failure mode
the compatibility matrix cannot see.
