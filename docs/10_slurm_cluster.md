# 10. Running on the watson cluster (Slurm)

## Cluster facts (measured 2026-11-11)

| item | value |
|---|---|
| host | `watson` (`ssh lujun.li@watson.uni.lux`) |
| scheduler | Slurm (`sbatch`/`srun`/`squeue`; there is no `swarm` binary) |
| partitions | `gpu` (default), `interactive` |
| nodes | `gpu1` (32 CPU, 192 GB, 1× RTX A6000 48GB), `gpu2`/`gpu3` (32 CPU, 64 GB, 1× A6000 each) |
| account | `sedanlab` |
| network | HF + PyPI reachable; **`fred.stlouisfed.org` times out (curl rc=28)** |
| python | 3.12.14 and 3.13.15 only; `uv python install` is forbidden (`python-downloads = "never"`) |
| working copy | `~/tsfm4finance` (clone of branch `bench/timesfm-spec41`) |
| HF cache | `~/hf_cache` (~27 GB, shared with other work) |

## Environment setup on the cluster

```bash
git clone -b bench/timesfm-spec41 https://github.com/DobricLilujun/TSFM4Finance.git ~/tsfm4finance
git clone https://github.com/NeoQuasar/Kronos ~/tsfm4finance/third_party/Kronos   # Kronos trainer
cd ~/tsfm4finance
export UV_PYTHON=3.12 HF_HOME=$HOME/hf_cache
for p in envs/*/; do uv sync --project "$p"; done          # tirex2 needs triton<3.6 -> 3.12 only
```

Two pin facts learned the hard way here:

* `requires-python` is `>=3.11,<3.13`: the cluster has no 3.11 and forbids downloading one,
  and the older families' `torch`/`triton` pins (2.3.1/2.4.1, triton 3.0/3.1) have **no cp313
  wheels**, so 3.13 is excluded and `UV_PYTHON=3.12` selects 3.12.
* `momentfm==0.1.4` **hard-pins `huggingface-hub==0.24.0`**, so `envs/moment` must not pin
  `huggingface-hub`/`numpy`/`pandas` itself — pinning `hub==0.36.2` (what the 3.11 laptop venv
  had force-installed) makes the whole environment unsatisfiable.

## Jobs

```bash
slurm/submit_train.sh fs16            # 24 series  x every family with a trainer (40 steps)
slurm/submit_train.sh fs64            # 12 long-history series (120 steps)
slurm/submit_train.sh full            # 12 long-history series, every train window, 3 epochs
sbatch --job-name=data-t234 slurm/data.sbatch                    # CPU-only tier prefetch + derive
sbatch --job-name=eval-ft --dependency=afterok:<full job ids> slurm/eval.sbatch
```

* one **array per model**, one **task per series** → Slurm spreads them as GPUs free, and a
  single series failing doesn't lose the model's other 11;
* `slurm/train_task.sh` maps model → family env (`envs/<fam>/.venv/bin/python`) and sets
  big-GPU knobs (`--bs 32 --ctx 512 --amp bf16` on the A6000 vs `--bs 4` on the 6 GB laptop);
* re-submitting is free: an existing checkpoint is skipped by the trainer, and the sweep is
  resumable per cell;
* data jobs request **no `--gres`** so they pack into idle CPUs of a gpu node.

Monitoring:

```bash
squeue -u $USER -o "%.10i %.24j %.9T %.8M %.10L"
tail -f logs/ft_<name>_<jobid>_<task>.log
ls reports/ft_ckpts | grep __full          # checkpoint inventory
envs/core/.venv/bin/python -m tsfm4finance.bench.sweep --stage status
```

## Status / known gaps at first submission

* submitted: `2571-2580` = `full` + `fs16` arrays for `chronos-2`, `ttm-r2`, `moirai-2.0-R-small`,
  `toto-2-22m`, `timesfm-2.5-200m` (180 tasks); `2581` = eval gated on the five `full` arrays;
  `2582` = GPU smoke for the four families not yet in the fleet; `2570` = data tiers (finished).
* all three GPUs were busy with other (`g27-*`) jobs at submission time, so the fleet queues.
* `envs/moment` failed to resolve because of PyPI connect timeouts here — retry with
  `UV_HTTP_TIMEOUT=600 uv sync --project envs/moment`.
* `moirai-1.1-R-base` trainer fails (`RuntimeError: The size of tensor a (16) must match ...`):
  the 1.1 predictor head has a different patch/quantile layout than 2.0; needs a fix before it
  can join the fleet.
* FRED (T1 macro) must be fetched from a machine that can reach `fred.stlouisfed.org` and
  rsynced in, or retried later.
