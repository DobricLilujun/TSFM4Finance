# trux probes (yewei@trux-tsubame.uni.lux)

Things established on this machine, which the `tsfm-trl` benchmark run still has to
absorb. Everything here was produced without the harness, because the harness code is on
another machine and **trux has no route to it** (laptop unreachable; DGX `10.6.32.13`
unreachable; `watson.uni.lux:22` times out from trux — the watson path only exists from
the laptop). The repo now has a remote, so the code can arrive by git:

```bash
git clone https://github.com/DobricLilujun/tsfm-trl.git /raid/yewei/projects_yewei/tsfm-trl
```

## machine

| | |
|---|---|
| GPUs | 2 × NVIDIA RTX 6000 Ada (48 GB each), driver 595.45.04, x86_64 |
| package tooling | `uv` at `/raid/yewei/.local/bin/uv`, `UV_PYTHON_DOWNLOADS=auto` needed (system config disables python downloads) |
| probe envs | `/raid/yewei/envs/tsfm-trl/<name>` — kept **outside** the future repo dir so a repo rsync cannot clobber them (`envs/local.json` points at absolute paths anyway) |
| HF cache | `/raid/yewei/.cache/huggingface` (447 GB) — already holds `amazon/chronos-2`, `amazon/chronos-bolt-base`, `google/timesfm-2.0-500m-pytorch`, `google/timesfm-2.5-200m-pytorch`, `google/timesfm-3.0-pytorch`; `Maple728/TimeMoE-50M` and the Tier-A set below were added by this session |
| HF token | account `etamin` (not the GitHub account) |
| **gated** | `Momonlp/MOMENT-1-small` / `-large` return HTTP 401 with this token → MOMENT rows cannot be produced on trux until a token with access is configured (`huggingface-cli login`). Not a model failure. |

Prefetched for the run: chronos-t5-small/base, chronos-bolt-small/tiny,
moirai-1.1-R-small/base, moirai-2.0-R-small, moirai-moe-1.0-R-base, ttm-r2, ttm-r3,
flowstate-r1, Toto-2.0-22m (`/tmp/prefetch.log`).

## TimeMoE-50M: three findings, one of them a correctness trap

Probe: `timemoe_probe.py` (this directory), env `/raid/yewei/envs/tsfm-trl/timemoe`
(python 3.11, torch 2.7.1+cu128). 24 rows in 169 s on one GPU; a batch of 8 windows at
h=126 takes ~5 s, so 48 GB is nowhere near a limit — the laptop's 6 GB OOM was machine
limits, not architecture.

**1. The checkpoint pins its own transformers version, and it is older than it looks.**

* transformers 4.53 → `AttributeError: 'list' object has no attribute 'shape'`, then
  `'DynamicCache' object has no attribute 'get_max_length'` (the remote code calls
  `past_key_values.get_max_length()`, gone in ≥4.42).
* transformers 4.41.2 → `RuntimeError: Tensors must have same number of dimensions: got
  3 and 2`: `_greedy_search` is deprecated there and routes to `_sample`, so the
  model's own `TSGenerationMixin._greedy_search` (which is the only code that knows the
  head emits a *continuous* `(B, horizon, input_size)` block, not token ids) never runs.
* **transformers 4.40.2 works.** `envs/envs.toml` should record exactly that reason, not
  just a version.

**2. The released 50M head is deterministic — there is no sample axis.** Two identical
rollouts differ by exactly `0.000e+00`, and `ts_generation_mixin.py` sets
`next_tokens = next_tokens_scores` (no argmax, no sampling). So any interval reported for
TimeMoE is **constructed by the caller** (residuals, an assumed error model), never
emitted by the model. In the matrix this belongs in `Pred.meta` and in the quantiles
column as a constructed band, not a native one.

**3. Feeding it raw price levels produces garbage that still runs.** MASE per series
(8 windows, ctx 256, local MASE implementation — comparable within this table only):

| series | h | raw-scale input | per-window z-norm input | naive MASE |
|---|---:|---:|---:|---:|
| sp500_daily (level ≈ 5·10³) | 21 | **169.0** | 4.43 | 2.82 |
| sp500_daily | 126 | **173.9** | 13.15 | 7.98 |
| btc_crypto (level ≈ 6·10⁴) | 21 | **58.7** | 3.19 | 2.82 |
| btc_crypto | 126 | **60.7** | 11.30 | 7.47 |
| audusd_fx (level ≈ 0.6) | 21 | 9.59 | 3.66 | 3.16 |
| de_10y_yield (level ≈ 2) | 21 | 3.70 | 3.87 | 4.33 |

The failure mode is silent: all outputs are finite, `predict` returns a tensor of the
right shape, and only the scale is wrong — the model answers in a normalised space while
the caller asked in price space. Everything at level ≫ 1 (equity indices, crypto) is
destroyed; everything at level ≈ 1 (yields, FX) survives by accident.

**Action for the harness:** check whether `model/adapters/timemoe.py` does per-context
scale/restore (subtract mean, divide by std, map the forecast back). If it does not, every
TimeMoE row in `EFFECTIVENESS.md` is inflated by this factor and the adapter needs the
normalisation, with the scale recorded in `Pred.meta`. A grep for the normalisation plus a
two-row compare (raw vs normalised, same window) settles it — and this probe is the
reference for what "wrong" looks like.

## datasets

`/raid/yewei/projects_yewei/TSFM4Finance/assets/data` is inventoried by
`scripts/inventory_datasets.py` in that repo → `docs/dataset_inventory.md` and
`reports/dataset_inventory.csv`: **132 usable datasets** (101 per-instrument `standard/`
series, 15 `open/` public finance series, 4 proprietary `closed/`, 12 `raw/` CSVs),
plus `assets/data/bis` — a curated BIS FX benchmark already built: 26 currencies,
238,742 observations, **50,632 evaluation instances**, lookbacks [365, 730], horizons
[5, 10, 20, 30, 60, 90], with Chronos-2 and TimesFM-3.0 zero-shot *and* fine-tuned
results and checkpoints in `bis/ckpts`.

That last item is the protocol the effectiveness table should grow into: pass 1 used 3
series × 12 windows × 1 seed. `standard/` (101 independent instruments) and `bis`
(50k instances, 26 currencies) are what make a horizon/skill claim survive a reviewer.

## MOMENT-1: the released checkpoint has no forecasting head

Weights obtained over git as asked (the registry id was wrong, not a gate):

```bash
git clone https://huggingface.co/AutonLab/MOMENT-1-small     # 359 MB
git clone https://huggingface.co/AutonLab/MOMENT-1-large     # 1.4 GB
```

`Momonlp/MOMENT-1-*` (what `models.yaml` points at) returns 401 for the trux token;
`AutonLab` is the real org and is public. Cloning over git also pins the commit the
weights came from. Env: `/raid/yewei/envs/tsfm-trl/moment`, frozen at
`/raid/yewei/envs/tsfm-trl/moment.trux.lock.txt` — `momentfm==0.1.4`,
`transformers==4.33.3`, `torch==2.7.1+cu128`. (`momentfm` ships no pandas/pyarrow, so
loading parquet data needs them added with `--no-deps` plus `pytz`/`python-dateutil`.)

`moment_probe.py` + `moment_head_probe.py`, small on cuda:0 and large on cuda:1
simultaneously. Everything below is measured, not read off the docs:

1. **The checkpoint contains exactly one head**: `head.linear` = (8, d_model) — the
   reconstruction head. `from_pretrained(task_name="forecasting")` does *not* change it;
   it loads that head and reports "0 missing keys" for every task, which looks like a
   clean load and is not one.
2. **`model.init()` — which the model card tells you to call — swaps the head and
   randomises it.** Shapes after `init()`: forecasting (21, d_model×64), classification
   (2, d_model), embedding (none), reconstruction (8, d_model, unchanged). The library
   says so in a warning: *"Only reconstruction head is pre-trained. Classification and
   forecasting heads must be fine-tuned."*
3. **`init()` also leaves the new head on CPU**, so a caller who does
   `.to(cuda)` → `.init()` gets a device-mismatch crash and may conclude the model is
   broken rather than that the call order is wrong.
4. **The head shape encodes two hard constraints**: fan-in `d_model × 64 patches` means
   the context must be **exactly 512 points** (its `seq_len`), and `forecast_horizon` is
   baked into the weights, so one instance serves one horizon. Measured: with ctx=256
   *every* configuration fails in the matmul (`mat1 (6×16384) and mat2 (32768×63)`); with
   ctx=512 everything runs.
5. **What an untrained head scores** (6 windows, ctx 512, `head_untrained: true`):

   | series | h | naive MASE | small, untrained head | large, untrained head |
   |---|---|---|---|---|
   | sp500_daily | 21 | 2.85 | 23.70 | 23.10 |
   | sp500_daily | 63 | 5.29 | 24.67 | 24.78 |
   | sp500_daily | 126 | 5.84 | 27.10 | 26.34 |
   | audusd_fx | 21 | 3.25 | 10.67 | 10.52 |
   | audusd_fx | 63 | 7.64 | 11.33 | 10.89 |
   | audusd_fx | 126 | 10.56 | 11.74 | 11.72 |

   10–27 MASE against a naive baseline of 3–10, and `large` is within 0.6 of `small`:
   parameter count does nothing for a head that was never trained. Any table reporting a
   "MOMENT zero-shot forecast" is reporting one of these numbers.
6. **What MOMENT *can* do zero-shot**: `embed()` → (B, 512) / (B, 1024) finite embeddings;
   `detect_anomalies()` → `reconstruction` and `anomaly_scores` of shape (1, 1, 256), both
   finite, both from pretrained weights. Classification by *probe on those embeddings* is
   legitimate; classification through the native head is not.

Consequences for the matrix: MOMENT's forecast cell is **declined for zero-shot**
(no pretrained forecasting head), with a real entry in the fine-tune column instead
(head training, optionally + LoRA on the backbone); its embedding and anomaly cells are
genuinely `ok`. An adapter must not call `init()` after loading weights, must pin ctx to
512, and must build one model per horizon.

Operational note: both cards are shared — other users were holding ~46 GB of 48 GB during
this run — so trux jobs need to be sized to the free memory, not to the card.

## Pass 2 on trux: stratified sample + lane runner

`sweep_stratified.txt` / `sweep_stratified.json` — 16 series for the expanded
effectiveness pass: 4 equity daily, 3 sovereign-yield daily, 3 FX daily, 3 FX hourly, plus
the three pass-1 anchors (`sp500_daily`, `btc_crypto`, `std_fx_1d_audusd`) so pass 2 stays
comparable to the published 384-row table.

The selection rule is fixed in the generator and deliberately independent of any model
result: within each stratum take the **longest** series (most evaluation windows),
tie-break by name. If a sample has to be small, it should be chosen by something that
cannot be result-shaped.

`trux_lane_runner.py` is a scheduler, not a second harness — all scoring stays in
`scripts/sweep_evaluate.py`. What it adds, each because it was observed here:

| behaviour | why |
|---|---|
| assigns lanes by **currently free** GPU memory, and refuses a GPU lane when no card has `--min-free-mib` (default 6000) free, downgrading it to CPU with a warning | measured on 2026-11-23: card 1 dropped from 16 GiB free to **644 MiB** inside one hour while four users' jobs ran at 99-100% utilisation |
| probes the env's `torch.cuda.is_available()` before launching a GPU lane and exits rather than producing rows nobody asked for | the DGX incident where three family envs silently resolved CPU-only torch |
| skips series that already have rows in a lane's JSONL | a 12-hour contended run will be interrupted; incremental JSONL must be restartable |
| writes `<lane>.env.jsonl` (free MiB, who else is on the card, loadavg) at lane start and exit | a slow row on a shared card is a property of the machine, so timings need their context recorded, not argued later |
| `--dry-run` prints the exact commands with no repo present | the flag spellings of `sweep_evaluate.py` get eyeballed before a run, not discovered by failure |

`_selftest/` runs the runner against a stub sweep script: it verifies lane splitting, the
free-memory downgrade, resume-skipping, and the contention sidecar without any weights.
Run it with:

```bash
python3 trux_lane_runner.py --repo _selftest/repo --env-root _selftest/envs \
    --series-file _selftest/three.txt --lanes cpu,cpu2 --min-free-mib 99999 \
    --out-dir _selftest/out
```

Cost proxy for splitting is currently `train.parquet` size / 4 — a guess, isolated in
`plan_lanes()` on purpose, and meant to be replaced by measured per-series seconds as
soon as a real pass produces them.

## Spark (DGX Spark / gpu7) as a dial-in worker

The Spark cannot be reached *from* trux (10.6.32.13:22 has no route, re-checked twice),
but it can ssh *into* trux with the dedicated transfer key. So trux is the control plane
and the Spark is a worker that dials in: `bridge/spark_worker.sh` polls, pulls the next
queued job, runs it, pushes the log back, and posts a heartbeat carrying its own
environment. No listener, no tunnel, nothing to open on the Spark.

```
trux:/raid/yewei/spark_bridge/{inbox,outbox}     <- control plane (this side)
gpu7:~/spark_bridge/                             <- worker's local mirror
```

Start the worker **once on the Spark** (from the laptop session, which can reach gpu7):

```bash
scp -i ~/.ssh/id_ed25519_gm_transfer \
    yewei@trux-tsubame.uni.lux:/raid/yewei/projects_yewei/trux_probes/bridge/spark_worker.sh \
    ~/spark_worker.sh
chmod +x ~/spark_worker.sh
REPO=$HOME/tsfm-trl PY=$HOME/envs/tsfm-trl/core/bin/python \
  nohup ~/spark_worker.sh > ~/spark_worker.log 2>&1 &
```

Then queue from trux: `bridge/trux_dispatch.sh bridge/spark_000_audit.sh --wait`.

Three properties the self-test was written to force (both bugs it found are real):

* **rsync transport *and* path prefix must switch together** — a "local mode" that left
  `user@host:` in the destination made the worker look alive while transferring nothing.
* **A job left in `inbox/` re-runs forever.** The worker now skips any job whose `.log`
  already exists, and `trux_dispatch.sh --wait` reaps the job into `inbox_done/`. After
  the fix: 5 polls, `runs_seen=1`.
* **The heartbeat carries provenance** (`host`, `arch`, `gpu_free`, `code_sha`, and the
  per-env `torch/CUDA` probe), because a cross-machine row without it is not a fact.

`pass2_trux.txt` (10 series) and `pass2_spark.txt` (6 series) partition the 16-series
sample with **no overlap**, and the split alternates *within each stratum* so that both
machians see every dataset family — otherwise "machine effect" and "dataset family" are
confounded and a cross-machine difference proves nothing. `spark_000_audit.sh` reports
the machine and probes every family env for CUDA *before* any row is trusted, and
`spark_100_sweep.sh` refuses to run if its `core` env has no CUDA (the DGX incident where
three envs silently resolved CPU-only torch).
