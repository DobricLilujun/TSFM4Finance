# 1 · Install

## Two installs, not one

```bash
# the library (any python 3.10–3.12)
uv venv envs/core && uv pip install --python envs/core/bin/python -e .

# the families that cannot share that interpreter
./scripts/make_envs.sh                       # every env declared in envs/envs.toml
./scripts/make_envs.sh core moirai moment    # or a subset
```

The library's own dependency floor is small:

```
numpy>=1.24 pandas>=2.0 scikit-learn>=1.3 scipy statsmodels>=0.14
pyyaml pyarrow huggingface_hub safetensors tqdm
```

Everything that *models* need lives in the per-family environments
(described in `envs/README.md`, machine-readable in
`envs/envs.toml`). `doctor --all-envs` runs each registry row in
its declared env, so a resolver fight between `uni2ts`, `momentfm`, `betatool` and
`neuralforecast` never happens.

## torch is platform-specific, so it is pinned separately

`make_envs.sh` installs torch from `TORCH_SPEC` / `TORCH_INDEX` instead of the lock:

```bash
# x86 + CUDA 12.x
./scripts/make_envs.sh core
# DGX Spark (aarch64, GB10, CUDA 13) — plain PyPI aarch64 wheels are already cu13
TORCH_SPEC=torch ./scripts/make_envs.sh core moirai moment timemoe tirex tirex2 kronos sundial
```

The locks (`envs/*.lock.txt`) are frozen from venvs where the matrix actually passed,
and are installed with `--no-deps`: their metadata is mutually contradictory
(`gluonts` asks for `pandas<2.2`; the working venv carries 2.3.3), and re-resolving
would refuse a configuration that is known to run.

## Weights

```bash
export TSFM_WEIGHTS_ROOT=$HOME/models
./scripts/prefetch_weights.py --tier A         # 20 rows; 5 rows have no public weights
```

`ModelHub` resolves a registry id in this order: `$TSFM_WEIGHTS_ROOT/<org>/<name>`,
then the matching directory anywhere under `$TSFM_WEIGHTS_ROOT`, then the Hugging Face
cache, then the Hub. Downloading during a measurement is why prefetching is a separate
step: a row that pulls 5 GB mid-run skews its timing and hides an offline failure as a
"model bug".

## Third-party checkouts (no wheel, code only in git)

Kronos, Sundial and VisionTS++ publish weights on the Hub but their code lives only in
a repository. Clone them anywhere and point the library at them:

```bash
git clone --depth 1 https://github.com/shiyu-coder/Kronos        $HOME/third_party/Kronos
git clone --depth 1 https://github.com/thuml/Sundial            $HOME/third_party/Sundial
git clone --depth 1 https://github.com/InfiniteZZZ/VisionTS     $HOME/third_party/VisionTS
export TSFMTRL_THIRD_PARTY=$HOME/third_party
```

`tsfm_trl/thirdparty.py` puts those directories (and their subdirectories, plus
`./third_party/*` and `./envs/*`) on `sys.path`. Note the orgs differ per family: the
**Kronos code** is `shiyu-coder/Kronos`, while its **weights** live under the
`NeoQuasar/` HF org. When an import fails, the error names the missing checkout and
prints the `git clone` line — a missing repo is reported as an environment gap, never
as a model failure.

## Verified platforms

| machine | what it proved |
|---|---|
| laptop, RTX Ada 6 GB, CUDA 12.x | development, core/moirai/moment/toto/tirex/timemoe/kronos/ttm envs |
| DGX Spark (GB10, aarch64, 119 GB unified) | the aarch64 matrix; TiRex-2's Triton path (sm ≥ 10.0, Triton ≥ 3.4) |
| 3× A6000 (escalation) | not needed so far: every Tier-A row fit on 6–48 GB at smoke size |

## The parquet engine is an environment dependency, not a model one

The canonical benchmark series are `train/validation/test` **parquet** directories, so
every family env that should be able to *read a dataset* needs a parquet engine. Two
pins collide: envs frozen on `numpy 1.26` cannot import `pyarrow ≥ 18` (`pyarrow
requires NumPy 2.0 or newer`), and pandas' fallback names `fastparquet`, which is not
what you should install into a frozen env. Use the version the numpy pin allows —

```bash
uv pip install --python envs/sundial/bin/python --no-deps pyarrow==16.1.0   # numpy 1.26 envs
uv pip install --python envs/core/bin/python     --no-deps pyarrow==26.0.1  # numpy 2 envs
```

`SeriesDataset._from_dir` now raises a `DataError` that says exactly this instead of
letting pandas' optional-dependency message read like a model failure.
