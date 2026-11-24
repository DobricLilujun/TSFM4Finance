# tsfm-trl

**TRL for Time-Series Foundation Models — a benchmark harness, not a training recipe.**
One API over every open time-series foundation model (TSFM), so a benchmark can treat 46
models as one thing: zero-shot forecasting (point + quantiles + sample paths),
classification at three levels, embedding probes, anomaly scoring — plus the capability
gates a benchmark row needs to be auditable (a training path, a reward path) and a
machine-checked compatibility matrix that says, per model and per task, what actually
works in *your* environment.

Nothing here is a wrapper around a paper's README. Every row of the matrix was produced by
running the code, and every `—` in it is a claim the code refuses to fake. If a family's
public build only exposes inference, the column says `— (no trainable module)` instead of
quietly optimising a non-differentiable quantity.

```bash
pip install -e .                      # core env; other families get their own venv
tsfm-trl list --tier A                # 46 registered models, 11 environments
tsfm-trl forecast --model chronos-2 --series my.csv --horizon 21
```

```python
from tsfm_trl import ModelHub, ForecastTrainer, TrainingConfig
from tsfm_trl.data.fixtures import tiny_finance

m = ModelHub.load("moirai-2.0-R-small", device="cuda")
pred = m.predict([tiny_finance().values[:256]], horizon=16)
pred.point, pred.quantiles          # (1,16) and (1,16,9) on the canonical LEVELS grid

r = ForecastTrainer(m, TrainingConfig(budget="fs16", peft="lora")).fit(tiny_finance())
r.status, r.loss_first, r.loss_last # 'smoke', 3.027, 3.111
```

## What it gives you

| | |
|---|---|
| **Models** | Chronos (2 / Bolt / T5), TimesFM (2.0 / 2.5 / 3.0), Moirai (1.1 / 2 / MoE), MOMENT-1, TTM-r2/r3, Toto, FlowState/Granite, TiRex & TiRex-2, Time-MoE, Sundial, Kronos, Kairos, Timestral, Hirondel, t0-α/β/γ, Yug, Aurora, VisionTS++, plus statistical baselines (naive, seasonal naive, theta, ARIMA, moving average) as sanity checks |
| **Benchmark tasks** | zero-shot forecast (point + quantiles + sample paths), classification (native head / LoRA head / frozen-embedding probe), embedding features, anomaly scoring — the same call signature for a diffusion head, a token LM and a flow-matching model |
| **Capability gates** | forecast fine-tuning (heads / LoRA / full) and reward post-training (grpo / rft / reward-descent): the point is that a benchmark can *adapt* a model and say exactly what was trained, not that adaptation wins |
| **Metrics** | CRPS, WQL, pinball, coverage (per emitted band), MASE, relative skill, ND, sMAPE, directional accuracy, rank correlation; PR-AUC / ROC-AUC / macro-F1 / F2-beta for classification (`docs/09_benchmark.md`) |
| **Honesty** | one canonical quantile grid (`LEVELS`), coverage measured only on bands the model actually emits, per-model `Pred.meta` recording *how* an interval was produced, and a `declined ≠ failed ≠ not-checked` vocabulary in the matrix |
| **Proof** | `tsfm-trl doctor` → `COMPATIBILITY.md`, one subprocess per family environment, reason strings captured for every non-✅ |

### Where the matrix stands

Two different claims, and the repo keeps them apart:

| | question | where the answer is |
|---|---|---|
| **compatibility** | does this architecture load, forecast, get scored, and have a training/RL path *in this environment* | `COMPATIBILITY.md`, produced by `tsfm-trl doctor` |
| **effectiveness** | how accurate is it, versus baselines and other families | `EFFECTIVENESS.md` (site: *Measured effectiveness*) — 43/46 models × 3 series × horizons 21/63/126, zero-shot, one seed; protocol and reading rules in [`docs/09_benchmark.md`](docs/09_benchmark.md) |

The first effectiveness pass is in the repo, and what it says is mostly about what
these models *cannot* do on finance data: on a daily index no published checkpoint beats
the trivial scaler on point accuracy at 21- or 126-day horizons, directional accuracy
sits at chance (~0.5), and the column that actually separates families is interval
calibration — where a nominal 80% band covers anywhere from 0.27 to 0.98 depending on
model and horizon. Rankings also move between series, so no single-dataset "best TSFM"
claim survives. The pass also caught an adapter bug the capability matrix could not see:
TiRex's roll-forward path crashed for any horizon past one output block, because the
matrix grid never reached that code.

`COMPATIBILITY.md` is the merged output of one run per machine — an x86_64 laptop
(RTX 1000 Ada, 6 GB) and an aarch64 DGX Spark (GB10, unified memory) — over all 46
registered models. Benchmark columns first, capability gates last:

| task | ok | declined by the model | failed | not checked |
|---|---|---|---|---|
| load | 45 | 0 | 1 (VisionTS++ ships `.ckpt` only) | 0 |
| forecast | 44 | 0 | 1 (TiRex-2 kernels, both machines) | 1 |
| quantiles | 35 | 9 | 0 | 2 |
| embedding probe features | 44 | 0 | 1 | 1 |
| classify (probe / head / native) | 44 | 0 | 1 | 1 |
| anomaly | 44 | 1 | 0 | 1 |
| reward post-training | 30 | 15 | 0 | 1 |
| fine-tune (capability gate) | 29 | 9 | 0 | 8 (statistical baselines, and models that never loaded) |

The failures are named in the file with their captured reason: TiRex-2 (Triton kernel
will not compile on either machine), VisionTS++ (no HF weights), and Chronos-T5-base on
the 6 GB card (an out-of-memory, which is why the same row is ✅ on the other machine).
The declined fine-tunes are TimesFM 2.0/3.0, TiRex, Yug, Aurora and Timestral (public
builds expose inference only) and PatchTST-FM (the installed build's `forward` declares
no target argument). "Not checked" is used literally: the task was not attempted for that
row, not refused.

Fine-tune went 20 → 29 ✅ in the last pass because three refusals turned out to be ours,
not the models' — Moirai 1.1/MoE train on uni2ts' own packed NLL, Sundial trains a
TimeFlow head rebuilt at the training patch length, and Kairos' per-window loss vector was
tripping our own finiteness check. t0 α/β/γ came along the same way (`_single_pass` is
differentiable and returns data-space quantiles). Fixing `peft="lora"` to actually freeze
the trunk also moved the numbers: what had been recorded as a LoRA budget was full
fine-tuning plus a few adapters (`docs/08_known_issues.md`).

### Which RL path each architecture exposes

The `reward` column carries the mode the *architecture* allowed — the trainer picks it,
the caller does not. `## RL reachability` at the bottom of `COMPATIBILITY.md` lists it per
model; the short version:

| path | what it needs | who can take it |
|---|---|---|
| `grpo` | sampled token paths **and** their log-probs (`sample_paths` + `path_logprob`) | Chronos-T5 (2 registered models) — the only true group-relative policy gradient available today |
| `descent` | a differentiable quantile head | Chronos-Bolt, FlowState, Moirai 2.0, Kronos-style heads… |
| `descent·pt` | only a point head | TTM r2/r3 |
| `rft` | anything that yields candidate futures (inverse-CDF or real samples) | Moirai 1.1/MoE, Time-MoE, Kronos, Toto, Kairos, t0, Hirondel — the majority |
| `—` | vendor build ships no trainable path | TimesFM, TiRex, Timestral, Yug, Aurora |

**Is the GRPO ours or a library's?** Ours — hand-written in
`tsfm_trl/training/reward.py` (~435 lines), with no RL library anywhere in the dependency
graph (`[rl]` extra = `torch` only; nothing imports `trl`/`trlx`/`verl`/`openrlhf`). The
*algorithm* is GRPO (Shao et al. 2024); this is a minimal implementation of it: group-
relative advantage with no critic ✓, but only the best-advantage path per group gets the
`−A·logp` term, no PPO clipping, and the KL is a weight-space L2 anchor to a frozen clone
of the starting parameters rather than a token-level KL to π_ref. Why not TRL's
`GRPOTrainer`: it is bound to a causal-LM interface, which excludes diffusion, quantile and
flow-matching heads — i.e. most of the field. Full discussion and the three steps to make
it canonical: [`docs/05_reward_rl.md`](docs/05_reward_rl.md).

## Quick tour

```bash
tsfm-trl list --tier A                        # 46 registered models, 11 environments
tsfm-trl info chronos-2                       # registry entry: caps, env, license, loss kind
tsfm-trl forecast --model tirex --series px.csv --horizon 21 --md

# the benchmark path: point + probabilistic metrics, one row per horizon
tsfm-trl evaluate --model chronos-2 --series px.csv --horizons 21,63,126 --ctx 512
tsfm-trl evaluate --model chronos-2 --adaptation reports/ft/chronos-2__…/model.pt \
                  --series px.csv --horizons 21,63          # zero-shot vs adapted

tsfm-trl classify --model moment-1-large --series px.csv --label dir5 --routes probe,head,native
tsfm-trl fit    --model moirai-2.0-R-small --series px.csv --peft lora --budget fs16
tsfm-trl reward --model chronos-t5-small --peft lora --steps 96 --bs 8 --lr 2e-5 --budget fs64
tsfm-trl ckpt   --ls reports/ft               # provenance of every checkpoint written

# compatibility, the way the published table is made
tsfm-trl doctor --all-envs --tasks load,predict,quantiles,embed,classify,anomaly \
                --json reports/matrix_bench.json                # benchmark columns only
tsfm-trl doctor --all-envs --tasks reward --json reports/matrix_reward.json  # fills the mode column
tsfm-trl doctor --merge reports/matrix_bench.json reports/matrix_reward.json \
                --json reports/compat_matrix.json --md COMPATIBILITY.md
```

`--series` is optional everywhere: without it the built-in tiny fixture runs, which is a
capability demo, not your data. `--md` no longer defaults to `COMPATIBILITY.md`, so a
single-environment run cannot overwrite the merged table (`docs/06_matrix.md`).

## Runnable examples

All four run in the `core` environment on one GPU with the bundled tiny fixture — no
weights beyond the core set, no dataset downloads:

```bash
python examples/01_forecast.py     # zero-shot forecast + CRPS/MASE/coverage table
python examples/02_finetune.py     # LoRA fine-tune, checkpoint provenance, reload checks
python examples/03_classify.py     # probe / head / native routes, train-vs-test gap
python examples/04_reward.py       # GRPO (token model) and reward descent (quantile head)
```

`toto-2-22m` (an RFT demonstrator) needs the `toto` environment.
`docs/05_reward_rl.md` carries every demonstrator curve with its parameter budget —
including the two that did **not** improve, kept as negative results.

## Documentation

| doc | |
|---|---|
| [`docs/01_install.md`](docs/01_install.md) | the eleven environments, why one interpreter cannot hold them, third-party checkouts |
| [`docs/02_forecast.md`](docs/02_forecast.md) | forecasting, the quantile contract, evaluation and metrics |
| [`docs/03_fine_tune.md`](docs/03_fine_tune.md) | training levels, LoRA budgets, checkpoints and provenance — and why fine-tuning is a *gate*, not the deliverable |
| [`docs/04_classify_probing.md`](docs/04_classify_probing.md) | the three classification levels, embedding probes, statistical probes |
| [`docs/05_reward_rl.md`](docs/05_reward_rl.md) | `RewardTrainer`: GRPO, reward-descent, rejection sampling — plus the provenance question (whose GRPO is this, and what is simplified) |
| [`docs/10_trl_api.md`](docs/10_trl_api.md) | the post-training API: reward definition, the four RL paths, the mode ladder, what an adapter must expose, measured demonstrators including the negative ones |
| [`docs/06_matrix.md`](docs/06_matrix.md) | reproducing `COMPATIBILITY.md`, how to read every symbol, benchmark-shaped vs full runs |
| [`docs/07_finance_tasks.md`](docs/07_finance_tasks.md) | price panels, `dir5`/`extreme` labels, anchor-relative metrics |
| [`docs/08_known_issues.md`](docs/08_known_issues.md) | the version archaeology: which pin breaks which family, and why |
| [`docs/09_benchmark.md`](docs/09_benchmark.md) | the benchmark protocol: metrics, splits, budgets, reporting rules, and what has *not* been measured yet |

### Documentation site

The same markdown, as a site (mkdocs-material → GitHub Pages):

```bash
pip install -r requirements-docs.txt
./scripts/build_docs.sh          # strict build into site/  (broken links/nav fail it)
./scripts/build_docs.sh serve    # http://127.0.0.1:8000
```

`mkdocs.yml` is the nav; `.github/workflows/docs.yml` publishes on push to `main` by
running **the same script**, which regenerates `docs/matrix.md` from
`reports/compat_matrix.json` — so the published compatibility table cannot drift away
from the machine-readable evidence. Enable Pages once per repo: **Settings → Pages →
Source: GitHub Actions**, or

```bash
gh api --method POST repos/$OWNER/$REPO/pages -f build_type=workflow
```

There is no second copy of the documentation to keep in sync: every page is a file in
`docs/`, and the site is optional — nothing about installing or benchmarking depends on
it.

## Status

* Machine-readable evidence in `reports/` (JSON, one row per model), rendered table in
  [`COMPATIBILITY.md`](COMPATIBILITY.md). Tier A = must work; Tier B = best effort;
  Tier C = documented-as-limited.
* Tests: `pytest -q tests` (40) covers the data/label layer, the metric and loss
  contracts, the registry (every model has an adapter, an env, a legal loss kind and caps
  that exist), the **CLI surface**, and the **evaluation layer**. Both of the last two
  were dead or wrong when they got coverage: `evaluate` and `info` crashed on attribute
  names nobody had ever run, and the covariate panel fed to panel-consuming models
  (Kronos) was sliced by a window's *position in a subsampled list* rather than its
  offset — a benchmark row would have been scored on the wrong covariates and still
  looked green.
* `./scripts/check.sh` runs pytest, `ruff` and a three-model mini-matrix, so "the library
  imports" is never the whole claim.

## Ground rules

* **No weights, not supported.** A model with no public checkpoint is out of scope
  (the registry keeps the row with `weights: null` and says so in the notes).
* **Small data by default.** Matrix rows and examples are smoke-sized (≤ a few hundred
  windows) on purpose: they prove *compatibility*. An effectiveness sweep is a different,
  longer run with its own protocol — `docs/09_benchmark.md` — and the budgets
  (`fs16`/`fs64`/`full`) exist so a smoke run can never accidentally become a benchmark
  run or vice versa.
* **Self-contained.** Adapters were ported from the TSFM4Finance benchmark, not
  imported from it.
* **Internal use.** Packaged as an installable distribution so publishing later is
  one command, but nothing is published now.

Apache-2.0 for this code; the model weights carry their own licenses (see the
registry and `docs/01_install.md`).
