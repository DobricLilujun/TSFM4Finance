# 6 · The compatibility matrix

```bash
tsfm-trl doctor --all-envs --md COMPATIBILITY.md --json reports/compat_matrix.json
```

One row per registry model, one column per capability. The point of the matrix is that
it is **produced by running**, so it can be re-run by anyone on their own hardware and
disagree with ours.

## What the table is for (and is not)

This is an **architecture-compatibility** table, not a results table. Column order says
so: the benchmark columns come first — `load · forecast · quantiles · embed · classify ·
anomaly` — then the two *capability gates*: `reward (mode)` and `fine-tune (capability)`.
A gate proves that a path exists and runs; it carries no accuracy claim. Effectiveness is
measured by `evaluate`/`forecast_eval` and documented separately in
[`docs/09_benchmark.md`](09_benchmark.md).

## How to read the symbols

| symbol | meaning |
|---|---|
| ✅ | it ran, and the output was checked (shapes finite, on the canonical grid, loss finite) |
| ✅ grpo / rft / descent / descent·pt | reward: **which RL path the architecture allowed** — token log-probs (grpo), sample-and-filter (rft), differentiable quantile head (descent), point head (descent·pt). The trainer chooses; see `docs/05_reward_rl.md` |
| ✅ (0.587→0.612) | fine-tune: first and last loss of the recorded curve. At matrix settings that is **two smoke steps**, so the pair proves the objective is differentiable and logging works — not that fine-tuning helps. A diverging run is still visible |
| ✅ 2/2 | classify: how many of the **three levels** (probe / head+LoRA / native) ran. `native` is only counted when the vendor ships a classification head at all, so MOMENT reads `3/3` and a quantile-head model `2/2` |
| — | the model/adapter **declines** this task, with a reason in the JSON (`skipped`) |
| `prepared` | a vendor loop was set up but not executed (`vendor_fit` paths, e.g. Chronos-T5's or Kronos' optional CSV script) |
| ◐ | partial: ran, but not fully verified or produced a degraded result |
| ❌ | it tried and failed; the captured reason is in the JSON |
| · | not checked in this run (`--tasks` subset) |

`skipped` and `failed` are deliberately different: the first is a statement about the
model, the second about the code or the machine. The `weights` column is separate from
`load`, because "we have the weights" and "the model loads here" are different claims.

## What a row actually does

`doctor.check(spec)` runs, in order: `load` → `predict` (+ quantile-grid checks) →
`finetune` → `classify` → `anomaly` → `reward`, each with its own try/except, and each
recording a status **and** a reason string. Data comes from `data.fixtures.tiny_finance`
(a synthetic regime-switching price series with an OHLCV panel) unless `--series` points
at a real dataset.

Sizes are fixed and small on purpose:

```
--ctx 256 --horizon 16 --tasks ... --budget fs16 --steps 2 (smoke) --reward-steps 6
```

so a row costs seconds, not GPU-hours, and every model is measured at the same scale.
`--clf-windows 192` caps the classification set the same way (evenly spaced, splits
proportional): a sampler model's classification features cost ~draws× more per window
than a quantile head's, and an uncapped feature pass turned one Kronos row into an hour.

## Benchmark-shaped runs

If the question is "which architectures can I benchmark here", leave the two capability
gates out and it is a few minutes per machine:

```bash
tsfm-trl doctor --all-envs --tasks load,predict,quantiles,embed,classify,anomaly \
                --ctx 512 --horizon 21 --json reports/matrix_bench.json
```

The reward pass is what fills the **mode** suffix in the `reward` column and the
`## RL reachability` table at the bottom of the markdown (which architecture exposes
token log-probs, which only sample-and-filter, which have a differentiable head, and
which vendor build simply ships no trainable path):

```bash
tsfm-trl doctor --all-envs --tasks reward --json reports/matrix_reward.json
```

Both write partial reports; merge them with the full one and the dedupe step keeps the
task coverage the partial runs produced (a reward-only re-run does not erase a model's
forecast column).

## Environment dispatch

`--all-envs` launches **one subprocess per environment** (`envs/<name>/bin/python`, plus
anything mapped in `envs/local.json`), passing `--env <name>` so each env only checks its
own rows. Reasons:

* a segfault, OOM-kill or CUDA abort in one family cannot take the run down;
* a family that needs `transformers==4.40.2` and one that needs 4.57.6 both run in the
  interpreter they were verified in;
* the JSON says which python and device produced each row.

An env that produces no JSON (argparse error, import crash, killed process) becomes a
row of its own — `<env tirex2> ❌ exit 1: …` — instead of silently contributing nothing.

## Two machines, one table

`--merge` combines per-machine JSON reports. By default it keeps **one row per
(model, environment)** — the row where the model actually ran — and writes the loser
into the winner's reasons under `other machine`, so an aarch64-only Triton failure
stays visible in an x86-clean table instead of vanishing. `--keep-both` opts out and
prints one row per machine; the markdown adds a `machine` column whenever more than
one machine contributed.

```bash
tsfm-trl doctor --merge reports/matrix_laptop.json reports/matrix_aarch64.json \
                --json reports/compat_matrix.json --md COMPATIBILITY.md
```

Useful flags:

```bash
tsfm-trl doctor --tier A --tasks load,predict --device cpu     # quick portability check
tsfm-trl doctor --model moment-1-large,moment-1-base --env moment
tsfm-trl doctor --merge reports/matrix_core.json reports/matrix_aarch64.json \
                --md COMPATIBILITY.md                          # combine per-machine runs
```

## Reproducing the published table

```bash
./scripts/make_envs.sh                                   # envs from envs/envs.toml
TSFM_WEIGHTS_ROOT=$HOME/models ./scripts/prefetch_weights.py --tier A
tsfm-trl doctor --all-envs --tasks load,predict,finetune,classify,anomaly,reward \
                --ctx 256 --horizon 16 --json reports/matrix_$(uname -m).json \
                --md reports/$(uname -m).md
```

Keep the JSON next to the markdown: the markdown shows the symbol, the JSON carries the
reason string, and a table without reasons is decoration.

`--md` defaults to **no file**, not to `COMPATIBILITY.md`. A single-env run used to
overwrite the merged table with its two rows; pass `--md` explicitly when that is what
you want (the `--merge` invocation still defaults to `COMPATIBILITY.md`).
