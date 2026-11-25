# tsfm-trl vendored into TSFM4Finance

This directory is the standalone `tsfm-trl` harness, vendored here so that every machine
that already clones `TSFM4Finance` gets it with one `git fetch`. Source of the copy:
`/home/ubadmin/projects/tsfm-trl` at commit
`4776f6e336b3a74923af0a44e31e0e93e0d4f905`. It was copied, not rewritten: adapters,
CLI, doctor matrix, tests and docs are the same files that produced `COMPATIBILITY.md`
and `EFFECTIVENESS.md`.

## What came across and what deliberately did not

| | |
|---|---|
| copied (3.0 MB) | `tsfm_trl/` (46-model registry, 24 adapters), `scripts/` (doctor, sweep_evaluate, reward_seeds, prefetch_weights, check), `tests/` (44 tests), `docs/`, `examples/`, `envs/*.lock.txt` + `envs.toml`, `COMPATIBILITY.md`, `EFFECTIVENESS.md`, `mkdocs.yml`, `probes/trux/` |
| evidence kept | 24 `reports/effectiveness_*.jsonl`, 8 `reports/reward_seeds_*.json`, 48 per-run fine-tune/reward config JSON |
| not copied | `envs/` virtualenvs (6.1 GB), `reports/ft*` + `reports/reward_demo` checkpoints (8.1 GB), `.venv-docs`, `site/`, `envs/local.json` (machine-local), `.github/workflows/docs.yml` (GitHub only reads the repository-root `.github`, so a nested one would be dead weight) |

The lock files are the point: rebuild the family envs from `trl/envs/<name>.lock.txt`
instead of reinstalling, or the benchmark is a different experiment.

## Branch policy

`trl-integration` is based on `origin/main` (`880a3bb`). The laptop's `main` has 10
unpushed commits (older TSFM4Finance benchmark work: watson slurm fleet, crypto data
layer, etc.) and is 35 commits behind upstream; **none of those were published** by this
change. To land it in the main working tree, rebase or merge this branch as usual.

## Verification at the moment of the move

Run against the copy, not the original, with `PYTHONPATH` pointing at the copy so the
editable install could not shadow it:

```
44 passed, 1 warning in 2.89s          # same count as the pre-move green state
tsfm-trl --help                        # CLI entry works from the copy
default_registry().models()            # 46 models load from the copy's models.yaml
```

Interpreter used for that check: `TSFM4Finance/.venv2` (python 3.11.15).

## Per-machine wiring

| machine | weights | third-party | envs |
|---|---|---|---|
| laptop | `TSFM_WEIGHTS_ROOT=$HOME/projects/TSFM4Finance/models` | `TSFMTRL_THIRD_PARTY=$HOME/projects/TSFM4Finance/third_party` | `envs/*/` already installed by the older project |
| trux | HF cache `/raid/yewei/.cache/huggingface` (452 GB) + `/raid/yewei/models` (MOMENT, cloned by git+LFS) | – | build from locks into `/raid/yewei/envs/tsfm-trl/<name>` (outside the repo dir so a sync cannot clobber them) |
| Spark (gpu7) | `~/models` | – | existing `~/envs/tsfm-trl/*` — **audit first**: `toto`, `timesfm20`, `new` resolved CPU-only torch (`2.7.0+cpu`/`2.8.0+cpu`) |
| watson | slurm node, needs `sbatch` (a login node has no GPU) | – | per-family envs as on laptop |

`probes/trux/` holds the machine-specific validation code: TimeMoE and MOMENT probes,
the GPU throughput probe, the stratified pass-2 sample (`sweep_stratified.*`,
`pass2_trux.txt` / `pass2_spark.txt`), the lane runner, and the Spark dial-in bridge.
Paths in those files are absolute trux paths on purpose — they record what was actually
run where.

## Open items carried over (not fixed by the move)

- **Corrected note (my own error, kept visible):** an earlier version of this file claimed
  `models.yaml` pointed MOMENT at `Momonlp/MOMENT-1-*` and that the 401 was a wrong-org
  bug in the registry. It is not: `models.yaml` already says `AutonLab/MOMENT-1-*`, which
  is public and works. The 401 came from probing the `Momonlp/*` mirrors mentioned in the
  older project, not from the registry. What *is* true: the released checkpoint carries
  only a reconstruction head (see `probes/trux/README.md`), the head pins context=512 and
  one horizon per instance, and `init()` after loading re-randomises it — all now recorded
  per row via `head_from_checkpoint` / `head_trained_here` in `Pred.meta`.
- Whether `tsfm_trl/model/adapters/timemoe.py` does per-context scale/restore: **checked
  in pass 2 — it does**. TimeMoE scored MASE 2.7–17, not the raw-level 60–169 the probe
  produced when levels were fed directly.
- TiRex-2 kernel JIT failure and the VisionTS++ packaging gap remain `failed`, not
  `declined`.
