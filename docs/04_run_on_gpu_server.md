# Running the GPU stages on another server

The whole sweep ran on a 6 GB RTX 1000 Ada, so it fits small GPUs — but the two
memory-hungry stages are `finetune` (full-parameter, 200M) and `valcheck` (it retrains to
score the fine-tuned model on the validation span). Anything bigger (24/40/80 GB) runs the
default settings comfortably.

## 1. Fresh server, one shot

```bash
git clone -b bench/timesfm-spec41 https://github.com/DobricLilujun/TSFM4Finance.git
cd TSFM4Finance
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[models,bench]"                    # timesfm + chronos + pmdarima + pyarrow
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA

# optional but recommended: pre-fetch weights so the run cannot stall on the network
huggingface-cli download google/timesfm-2.5-200m-pytorch
huggingface-cli download amazon/chronos-2

nvidia-smi                                          # sanity check
python -m tsfm4finance.bench.run                    # full sweep, incremental/resumable
```

Datasets are in the repo (`data/open/**`, including the 48 generated task datasets), so no
download step is needed. Re-running any stage skips finished work
(`(series, horizon, model)` keys already in `reports/timesfm_benchmark_results.json`).

## 2. Fitting the fine-tune into a small GPU

```bash
# bf16 activations (~half the activation memory) + smaller batch, shorter context
python -m tsfm4finance.bench.run --stage finetune --ft-amp bf16 --ft-bs 4 --ft-context 192
python -m tsfm4finance.bench.run --stage forecast --ft-context 192   # must match for comparability
python -m tsfm4finance.bench.run --stage valcheck  --ft-amp bf16 --ft-bs 4 --ft-context 192
```

Knobs: `--ft-bs` (batch), `--ft-context` (window length; the matched-context zero-shot run
is `timesfm@<same>`), `--ft-amp fp32|bf16|fp16` (loss stays fp32), `--ft-lr`,
`--ft-epochs`, `--ft-device cuda:0`. OOM inside `finetune()` already retries with half the
batch. Loss is pinball over 0.1–0.9 plus an L1 term on the median channel — unchanged.

For a partial sweep while you debug: `--series sp500_daily,btc_crypto`,
`--horizons 5,21`, `--contexts 128,256,512`, `--limit 2`.

## 3. Do not run two sweeps on one GPU

The harness holds a single-writer lock (`reports/.bench.lock`); a second process exits with

```
another benchmark run is alive (pid NNN); wait for it, or pass --force ...
```

Two concurrent runs on one 6 GB card OOM each other (this happened once and is the reason
the lock exists). If a run died and left the lock behind, it is auto-cleared when the
recorded pid is dead; `--force` overrides.

Also set these for stability on shared/fragmented GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. Disk discipline

Each 200M checkpoint is ~700 MB. The harness deletes a checkpoint as soon as the rows that
needed it exist (`_gc_checkpoints` after each forecast row; per-series reclaim inside
`trading`/`valcheck`), and `finetune` retrains only while a consumer still needs the
weights. Steady-state peak is ≤ 6 checkpoints (~4 GB). If your disk is tight:

```bash
for L in 2 4 6 8 10 12; do
  python -m tsfm4finance.bench.run --stage finetune --limit $L
  python -m tsfm4finance.bench.run --stage forecast --limit $L
  python -m tsfm4finance.bench.run --stage trading  --limit $L
done
python -m tsfm4finance.bench.run --stage valcheck --limit 12
python -m tsfm4finance.bench.run --stage derive --stage leaderboard --stage report
```

## 5. Bring results back and refresh the paper

```bash
rsync -av you@server:~/TSFM4Finance/reports/ ~/projects/TSFM4Finance/reports/
cd ~/projects/TSFM4Finance
python -m pytest tests/test_bench.py -q            # completion invariants must pass
python -m tsfm4finance.bench.paper_table --apply   # rewrites the survey block from the JSON
cd ~/projects/TSFM_Paper && ./build.sh && git commit -am "refresh §4.4 numbers" && git push
```

`paper_table.py --apply` writes between the `TSFM-BENCHMARK-AUTO` markers in
`chapters/4_datasets.tex`, so the paper can never drift from the results JSON.
