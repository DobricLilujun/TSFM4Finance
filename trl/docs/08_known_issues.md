# 8 · Known issues, pins and the reasons behind them

Everything here was measured, not read. Where a family is broken, the entry says what
the library does about it: **fix**, **work around**, or **report honestly**.

## The pins that decide whether a family runs

| family | pin | what happens if you ignore it |
|---|---|---|
| Time-MoE | **`transformers==4.40.2`**, `numpy<2` | 4.42 introduced `Cache` objects → `DynamicCache has no attribute from_legacy_cache`; 4.44 removed `standardize_cache_format` → the released remote code dies. 4.38 is too old (`EosTokenCriteria` missing). Only 4.40.x works. |
| TTM-r1 | `transformers==4.44.0` | 4.5x removed `PatchTSMEncoderLayer`. r2/r3 are fine on 4.57, which is why they live in `core`. |
| Sundial | `transformers==4.42.4`, `numpy<2`, `einops==0.8.1` | its repo code predates the 4.5x generation refactor. |
| peft everywhere | **`peft==0.19.1`** | 0.20.x removed the `base_model` keyword our LoRA path passes. |
| Toto | `safetensors==0.4.5` | 0.4.6 changed the deserializer its loader expects. |
| TimesFM 2.0-500M | **`timesfm==1.2.9` + jax** (env `timesfm20`) | PyPI's `timesfm` 2.0.x and 3.x are the **2.5/3.0** models; only 1.2.x still ships the v1 API (`TimesFm`, `TimesFmHparams`). 1.2.x imports `jax` at package import even for the torch backend. |
| TiRex-2 | Triton ≥ 3.4 **and** sm ≥ 10.0, else a CUDA toolchain (nvcc) | Triton kernels die on older archs; the *native* fallback JIT-compiles a CUDA extension through `flashrnn`, so it needs `nvcc` — a runtime-only CUDA install (no toolchain) fails with `Ninja is required…`. On a laptop that is a machine gap, and `TiRex2` raises `UnavailableError` naming both requirements rather than pretending the model is broken. |
| datasets | `datasets>=3,<4` | 4.x needs an ABI-matched pyarrow. |

## Model-level behaviour the adapters encode

* **TimesFM 2.0-500M is 50 layers × 1280 dims × 16 heads.** Building it with the 2.5
  shape loads a state_dict with 700 "unexpected keys". A mismatch is reported as
  *architecture*, with the layer count, not a key dump. Forecast comes from
  `TimesFm.forecast(inputs) -> (point, quantiles)`; there is no public training code,
  so fine-tune and reward are `—`.
* **Time-MoE's `generate(do_sample=True)` is broken in the released remote code**
  (its `prepare_inputs_for_generation` hands transformers a 3-D tensor and `_sample`
  unpacks 2-D). Greedy works; sampling does not, in every transformers we tested.
  The adapter therefore does its own AR decode with a KV cache
  (`_ar_decode`), and the sampled paths are recorded in `Pred.meta`.
* **Time-MoE's `forward` mutates the tensor you pass it** (in-place `unsqueeze`), so
  the decode loop clones on entry — otherwise the second step dies on a shape error
  that blames your own tensor.
* **Moirai 1.1 / Moirai-MoE cannot be fine-tuned through this API.** Their `forward()`
  *draws samples* from the predicted distribution and `_get_distr` returns a
  distribution over the whole padded window, not the prediction slice: a gradient
  through "the quantiles of a sample" would be a no-op dressed as a fine-tune. The
  honest answer is `skipped`, with the reason, and the note that a real fine-tune
  needs uni2ts' own Lightning stack (`MoiraiForecast` + packed masked dataset).
  Moirai **2.0** exposes a differentiable quantile head → fine-tune ✅.
* **MOMENT fixes `seq_len`/`pred_len` at `init()`.** Trainers call
  `prepare_training(ctx, horizon)` before asking for the module, and pipelines are
  cached per `(task, horizon, seq_len)` — rebuilding a 1 B backbone per call would
  dominate any fine-tune. MOMENT also turns gradient checkpointing on by default and
  its recomputation is not deterministic (`different number of tensors was saved`);
  the adapter disables it.
* **MOMENT has no quantile head** (`forward` returns mean + samples), so reward
  post-training uses the `reward_descent_point` path (scale-free accuracy + direction),
  with the interval terms switched off rather than silently reported as zero.
* **FlowState/Granite**: `prediction_length = -1` in the config means *flexible*; the
  adapter discovers the real chunk length with one dry forward (`_probe_head_len`).
  Vendor models keep tensors in plain attributes, so LoRA/offload uses
  `peft.move_deep` to walk non-registered tensor buffers. CUDA toolchain failures
  (nvrtc, arch mismatch) are treated as environment, and the trainer retries on CPU.
* **Chronos-T5** tokenizes on CPU and moves the ids; label windows are truncated to
  `config.prediction_length`.
* **TiRex's loader validates a model *id*, not a path** (`Invalid model id` when it is
  given a directory). `ModelHub` therefore passes `source_id=` (the registry id) to
  adapters so resolution to a local directory does not destroy the id.
* **Kronos needs an OHLCV panel**, not a target series: `SeriesDataset.panel_window`
  re-attaches the target column (its `close`) because the panel drops it. A 1-D series
  is still accepted: the adapter writes the flat panel the vendor's own `finetune_csv`
  format uses (`open = high = low = close`, `volume = amount = 0`) — that is their
  convention for a single price series, not an invention here.
* **Kronos' `predictor.predict` returns a DataFrame of six OHLCV columns, averaged over
  its internal samples** — so `sample_count` decodes happen and then the spread is
  thrown away. The adapter draws them itself through `predict_batch` (series × draws in
  one batched decode; 32 sequential calls turned a matrix run into an hour) and keeps
  `Pred.samples`, which is the only honest interval a sampler has.
* **Kronos fine-tunes in-process**: `training_step` is the vendor's own next-token
  cross-entropy (frozen `KronosTokenizer.encode` → `head.compute_loss`), so LoRA, the
  memory ladder and the loss curve apply like every other family. `vendor_fit` still
  exports their CSV + config and prints the command for anyone who wants the script.
* **Kairos ships weights but no remote code** (`architectures: ["KairosModel"]`, no
  `.py` in the repo). It needs `pip install git+https://github.com/foundation-model-research/Kairos`
  which provides `tsfm.model.kairos.AutoModel`.
* **TTM r2/r3 declare no quantile levels.** Their configs carry
  `distribution_output=student_t` and no `quantiles` list, so the head emits
  distribution *parameters*; inventing a 9-level grid from the parameter count would
  print an interval the model never predicted. `_load_keep_info` keeps the
  `from_pretrained` loading report, and where head tensors are **newly initialised**
  (ttm-r3 ships 80 of them that its released config does not describe) the adapter
  *removes* the `quantiles` capability and says why. Those rows are point-only, and
  reward post-training uses `reward_descent_point`.
* **`f1_score` does not take a `beta`.** The recall-weighted F2 the finance framing
  needs lives in `fbeta_score`; passing `beta=` to `f1_score` raised and was swallowed
  by the except, so `f1_beta2` was silently `nan` for every run. For >2 classes it is
  now the macro average of each occurring class's one-vs-rest F2.
* **PR-AUC and ROC-AUC need scores, not argmax.** `classification_report` computes them
  one-vs-rest (weighted) when handed the score matrix, and reports `nan` — not a
  number — when the caller only has hard labels. All three classification routes now
  pass softmax scores.
* **A from-scratch head needs a from-scratch learning rate.** The budget lr (1e-5) is a
  fine-tuning rate; at that step size a fresh MLP never leaves its initial bias, which
  reads as accuracy *below* chance. `fit_head` uses 1e-3 for the head with the trunk
  still on the fine-tuning rate (AdamW param groups).
* **stdout is a data channel.** `StepLog` writes its JSON lines to stderr, and the
  per-env matrix children write their report to a temp file: a parent parsing a
  child's stdout for "the JSON" silently dropped every row once the step lines started
  sharing the stream.
* **statsmodels never shipped `ThetaModel`** (absent in every release through 0.14/0.15,
  promised since 0.12). The theta baseline is implemented in-house (SES + drift).

## Design decisions that look like quirks

* `CAPS` is a **frozenset** — `set(caps) <= CAPS` in the registry tests is a subset
  check, and a tuple would make it silently wrong.
* `quantile_coverage` reports **only the bands a model actually emits**. Reporting 95 %
  coverage for a model whose grid stops at 0.9 is an interval lie.
* `Pred.meta` always says how an interval was produced (native head, empirical from
  *n* sampled paths, temperature-sampled AR decode, median-rollback rollout for
  horizons longer than one output patch).
* `skipped` ≠ `failed`: `—` means the model/adapter declines the task (with a reason),
  `❌` means it tried and broke.
* `smoke` counts as ok: the matrix checks that a capability *runs*, not that training
  converged. Loss curves are still printed (`✅ (0.587→0.612)`) so a fine-tune that
  diverges is visible.

## Vendor packages: the exact installs

These families publish no wheel on the index their model card names, and the matrix
rows say which one is missing. On the laptop they live in `.venv-vendor` / `.venv-toto`
(registered in `envs/local.json`):

| family | install | import |
|---|---|---|
| t0-α/β/γ | `pip install "tfc-t0[torch]>=0.6.1"` | `from t0 import T0Forecaster` (it *is* the nn.Module) |
| Timestral 5/28/123/209M | `pip install git+https://github.com/mistralai/timestral.git` | `from timestral import TimestralForecaster` |
| Hirondel-7M | `pip install git+https://github.com/tncrd/hirondel.git` | `HirondelForecaster.from_pretrained(...).forecast([...])` |
| Yug | `pip install yug` | `from yug import YugPipeline` |
| Aurora | `pip install aurora-model` | `from aurora import load_model` |
| Toto 2.0 | `pip install toto-models` | `from toto2.model import Toto2Model` |
| Toto 1.0 | `pip install toto-ts` (+ `setuptools<81` for `pkg_resources`) | `from toto.model.toto import Toto` |

* **Vendor forecaster objects keep outputs on the GPU.** `np.asarray` on a CUDA tensor
  raises `can't convert cuda:0 device type tensor to numpy`, which the matrix used to
  file as a failure of the model; `base.to_numpy` handles torch/jax/pandas.
* **Hirondel's forecaster is constructed from a model object**, not a repo id:
  `HirondelForecaster.from_pretrained(repo)` then `.forecast(list_of_contexts,
  prediction_length=)` → `.median (N,H)`, `.quantiles (N,H,L)`, `.quantile_levels`.
* **Toto-Open-1.0 (`toto-ts`) is the one row still open.** Its `TotoForecaster` takes
  the *backbone* (`Toto(...).model`) and a `MaskedTimeseries(series, padding_mask,
  id_mask, timestamp_seconds, time_interval_seconds, num_exogenous_variables)` built by
  their own dataset pipeline; the naive construction reaches `einops` shape errors.
  The adapter wiring is pending and the row reports that, rather than scoring an
  unverified path. Toto **2.0** (22m/313m) — the current generation — loads, forecasts,
  classifies and fine-tunes.
* **`~/.cache/huggingface/modules/transformers_modules` owned by root** (someone ran a
  pipeline as root on that box) makes every `trust_remote_code` model fail with
  `PermissionError`. Fix without sudo: `export HF_MODULES_CACHE=$PWD/.cache/hf_modules`.

## Numerical ladder

A non-finite loss on step 1 is a property of *that profile*, not of the family. The
fine-tune ladder now runs `keep → bf16 → fp16 → fp32 → lr×0.1 → ctx/2 → bs=1 → CPU`,
and `_is_unstable` (nan/inf in the message) admits a run to the ladder instead of
returning `failed` — which is how Toto-2.0-313m trains.

## Out of scope (and why)

* **Moirai 2.0-R-base / -large**: the repositories do not exist (404), gated or not.
* **Yug-1 (commercial)**: no weights. `birlaailabs/yug` (CC-BY-NC) is registered instead,
  with the license recorded in the row.
* **VisionTS++ / t0-α/β/γ base weights**: the authors host them outside the Hub; the
  rows carry `weights: null` plus a pointer instead of pretending they are fetchable.
* Any model whose weights require a login we do not have.

## Second pass: per-machine matrix, three classify levels

The matrix is now run on two machines (x86 laptop, RTX 1000 Ada 6 GB; aarch64 DGX Spark,
GB10 unified memory) and merged. What that surfaced:

* **One row per (model, env), the working row wins.** When the two machines disagree —
  TiRex-2 compiles on sm8.9 but its Triton kernel raises `CompilationError` on the GB10;
  MOMENT-large fine-tunes on 6 GB fp16 but OOMs then recovers on unified memory —
  `dedupe()` keeps the row that ran and records the other machine's reason under
  `other machine:`, so a failure never disappears, it moves to a footnote. `--keep-both`
  opts out. A row whose `env` no longer matches the registry is dropped at merge time:
  the model moved environments, and listing it twice double-counts it.
* **Machine gap ≠ model failure.** `UnavailableError: needs Triton and an sm10.0+ GPU, or
  nvcc for the MLSTM fallback` is an *environment* statement. The same model is ✅ on the
  other machine. The matrix prints both columns; the prose in this file is where the
  difference is explained.
* **`trust_remote_code` on a shared box**: `HF_MODULES_CACHE` must point somewhere
  writable, or every custom-code model (Chronos-Bolt variants, VisionTS, Sundial, Kairos)
  fails identically at import time.
* **Vendor envs are architecture-specific.** A `freeze.txt` generated on x86_64 cannot be
  installed on aarch64. On the DGX the Toto env had to be rebuilt from the *local* core
  freeze plus the Toto delta (`transformers==5.10.1`, `safetensors==0.4.5`,
  `tokenizers==0.22.2`, `torchmetrics`, `setuptools<81`); with the frozen x86 file it
  silently ends up without `transformers` at all and every row says `missing package`.
* **TimesFM 2.0/2.5 are v1-API models** (`timesfm==1.2.9` + JAX). The PyPI package's 2.x
  and 3.x lines are different checkpoints with a different API; the adapter pins the arch
  dict (`num_layers=50, num_heads=16, model_dims=1280, input_patch_len=32,
  output_patch_len=128`) because the config in the repo doesn't carry it.
* **Aurora's `load_model()` returns a model in train mode**, so its norm layers raise
  `Expected more than 1 value per channel` on the batch of one that inference is made of;
  the adapter calls `.eval()`. Its flow-matching head has no public training path, so
  fine-tune and reward report `—`, not `❌`.
* **Hirondel fine-tunes in its own normalised space.** `HirondelModel.forward(x,
  observed, horizon)` returns quantiles after mean/sd + `asinh` scaling; the adapter's
  `training_step` applies `_scale` to the context and the same transform to the target
  before the pinball loss, so the optimised objective is the one the forecaster inverts.
  Reward runs as `rft` (its head is not a differentiable quantile head in the sense
  `reward_descent` needs).
* **Kairos emits 128 steps for a 16-step request** (its `prediction_length` is fixed by
  the checkpoint). The adapter slices; fine-tune remains open because the vendor forward
  branches on tensor truthiness — reported as `❌ Boolean value of Tensor ...`, not hidden.
* **MOMENT's native classification head** takes `num_class` (not `n_class`), and its
  pipeline `forward` is keyword-only (`x_enc=`). Both appear in `fit_native`, which now
  tries positional then keyword. The pipeline object is cached: calling `pipeline()` per
  step re-downloads.
* **Reward: `--steps` beats `reward_steps`.** `reward_steps` is the *matrix* cap (a
  reward row checks that post-training runs, in seconds). Silently trimming a requested
  48-step demonstrator to 6 was how a run "did not improve". The cap now applies only
  when no step count was given.
* **Reward trend is reported by quartile means** (`reward_early_mean` vs
  `reward_late_mean`), not first step vs last step. First-vs-last called a run that
  improved (1.72 → 1.83 across quartiles) a regression, because step 1 happened to be a
  lucky batch.

## Third pass: Sundial, Toto 1.0, and what a moving repository does to a matrix

* **Sundial's Hub repo changed under us.** The current revision has no
  `prepare_input_ids`, ships **no tokenizer files**, and its model card's Quickstart is
  the truth: `model.generate(torch.randn(B, lookback), max_new_tokens=H, num_samples=k)`
  → `(B, k, H)` de-normalised samples (the mixin does revin itself). The older
  `prepare_input_ids(X=…, system="forecast")` + `tokenizer=` call is still in a lot of
  second-hand examples and no longer exists; the adapter prefers whichever API the
  loaded code provides. Two version facts, both verified on the matrix machine:
  `transformers==4.40.1` is required (4.42.x mis-builds the causal mask for patched
  inputs and `generate` dies with a 256-vs-16 shape error), and `device_map=` must not be
  passed to this remote code (accelerate re-dispatches through the Auto mapping, which
  does not know `model_type: sundial`, and raises "Unrecognized configuration class").
* **Sundial fine-tune is declined, not failed.** `forward(labels=…)` reaches their
  TimeFlow loss, whose packing (`output_token_lens`, `diffusion_batch_mul`, `mask_y`) is
  set by the training scripts rather than the checkpoint config. Calling it with the
  checkpoint's own config fails on shape (`addmm: (8x17664) @ (768x768)`). Writing an
  invented objective to turn that row green is exactly what this repository is not for,
  so the row says `—` and points at `thuml/Sundial`'s `scripts/`.
* **Toto-Open-1.0 does not go through `transformers` at all** — the repo has no remote
  code and no `model_type`, so the Auto path raises "Should have a `model_type` key".
  The vendor package `toto-ts` is the implementation:
  `Toto.from_pretrained(repo, map_location=...)` (a `ModelHubMixin` loader — no
  `device_map`/`dtype` kwargs, they get forwarded into the model config and crash the
  constructor), then `TotoForecaster(model=model.model)` — it wants the **backbone** —
  and a `MaskedTimeseries` built exactly like their own `evaluation/fev` pipeline
  (relative timestamps, `time_interval_seconds=1`, `id_mask` ones). `num_samples` must be
  divisible by `samples_per_batch`; intervals are empirical quantiles of the sampled
  paths and `Pred.meta` says so. Fine-tune uses their `CombinedLoss` (NLL + general
  robust loss) on the causally-scaled window, which is the pretraining objective.
* **PatchTST-FM (tsfm_public) declares no target argument** in the installed build
  (`forward(past_values, past_observed_mask, pad_mask, prediction_length, quantile_levels,
  return_loss)`), so a fine-tune that passed `future_values=` used to be a `TypeError`.
  The adapter now picks the target kwarg from the signature and, when there is none,
  declines with the signature in the reason. Its head is `backbone.out_layer`, so
  `head_prefixes` had to be declared — `heads_only` (the reward ladder profile) was
  failing with "matched no parameters" against generic prefixes.
* **TiRex-2 fails on both machines, for different reasons**, and the matrix says so:
  x86 (sm8.9, no nvcc) → neither Triton nor the CUDA fallback is available
  (`UnavailableError`); aarch64 GB10 → Triton *starts* compiling their recurrent kernel
  and raises `CompilationError` inside it. It is the only registered model that does not
  forecast anywhere here, and it stays ❌ rather than being dropped from the table.
* **`tsfm_trl` now fixes the HF dynamic-module cache at import** if
  `~/.cache/huggingface/modules` is not writable (someone ran a pipeline as root on the
  box). Without it, *every* `trust_remote_code` model fails with `PermissionError`, which
  a compatibility matrix would file as 6 model failures. The redirect prints one line to
  stderr rather than happening silently.

## Fourth pass: three fine-tune refusals that turned out to be ours

The fine-tune column carried `—` for Moirai 1.1/MoE and Sundial, and ❌ for Kairos. All
three were harness bugs wearing a model's face; each is now ✅, and the fix in each case
was to stop inventing a step and go read what the vendor's own training does.

* **Moirai 1.1 / MoE** — `forward()` returns a *distribution*; `predict` samples from it.
  Because the sampled path is gradient-free, the adapter declared "no differentiable
  head". The supervised step was in the same class the whole time: `MoiraiForecast` builds
  packed fields (`target/observed_mask/sample_id/time_id/variate_id/prediction_mask`) and
  scores them with `PackedNLLLoss` — uni2ts' own `MoiraiFinetune` default. The adapter now
  calls `_convert(...)` with the true future as `future_target` and optimises that.
  Trainable: 4.1 M (`heads_only`) / 5.9 M (LoRA) on `moirai-1.1-R-small`, 13.5 M on the
  base, 7.2 M on MoE. `loss: vendor` in the registry now says what the objective is.
* **Sundial** — supervised `forward(labels=…)` with the shipped config fails on shape
  because `output_token_lens=[720]` is a *generation* setting: labels get trimmed to
  `seq_len − input_token_len + 720`, a length a short window never reaches. Their
  fine-tune config sets the output token length per run; `prepare_training` now does the
  same and rebuilds `flow_loss.FlowLoss(16, 768, 3, 768, 50)` at `input_token_len`.
  Cost, stated in the adapter docstring: that head is **re-initialised** (11.4 M params),
  the 12-layer trunk transfers. Frozen trunk + new head on 12 steps: 3.77 → 0.25.
  `peft.lora_trains_head` exists for exactly this case — a rebuilt head has no pretrained
  weights for LoRA to nudge, so it trains outright next to the trunk adapters.
* **Kairos** — `RuntimeError: Boolean value of Tensor with more than one value is
  ambiguous` looked like a gradient-unsafe vendor forward. It was our own
  `if not torch.isfinite(loss)` running on the **per-window loss vector** their supervised
  forward returns. The trainer now reduces non-scalar losses to their mean (and says so in
  the run notes); 10 M/50 M train at 2.97 M / 7.17 M LoRA parameters.

One more honesty note that came out of this pass: the loss *curve* of a 2-step smoke says
nothing — `smoke=True` caps the run at two steps, and Moirai's packed NLL swings ±0.9
between random batches. The matrix ✅ means "a differentiable vendor objective ran and
logged"; direction needs ≥20 steps (`docs/03_fine_tune.md`).

## `peft="lora"` used to include the trunk

Worth its own heading because it inflates every LoRA number recorded before the fix:
`apply_lora._inject` froze the `nn.Linear`s it wrapped and enabled their adapters, but
froze nothing else, and HF models load with `requires_grad=True`. `make_trainable` now
freezes all parameters before injecting, then re-enables the adapters (and, for adapters
that *re-initialise* a head — `lora_trains_head`, used by Sundial — that head). Two
symptoms that were easy to misread as model behaviour: t0-gamma's 156 M "trainable"
parameters, and a t0 LoRA loss that rose from 2.2 to 6.0 in two steps because the trunk
was being moved at the fine-tuning rate.

Related: the `lora_trains_head` branch called a helper that did not exist
(`_matches_any` instead of `_match`), which is a NameError waiting inside a path only
Sundial takes. `ruff check tsfm_trl --select F,E9` catches it; `check.sh` runs that
before the matrix, which is exactly why `check.sh` exists.

## Fifth pass: two dead CLI paths, and Chronos-2's output-patch cap

* **`tsfm-trl evaluate` and `tsfm-trl info` were both broken and untested.** `evaluate`
  called `_load_model(a.model)` where the function wants the whole namespace
  (`'str' object has no attribute 'model'`); `info` printed `s.note` while the registry
  field is `notes`. Neither path is used by the matrix or the examples, so 32 green tests
  said nothing about them. `tests/test_cli.py` now drives the CLI surface on the
  statistical baselines — no weights, no GPU, two seconds — which is exactly the layer
  that had no coverage.
* **Chronos-2's supervised forward is capped by `num_output_patches`, default 1.** The
  adapter called `model(context=…, future_target=…)` without it, so any horizon above one
  output patch (16 steps) died in `_validate_input`, and the channels-last retry below it
  turned that into the still more misleading `context must have shape
  (batch_size, context_length)`. The vendor derives the argument from the prediction
  length; the adapter does the same now (`ceil(H / output_patch_size)`), so
  fine-tune/reward work at 16/21/64. Nothing about the checkpoint is rebuilt — the output
  embedding is applied per patch.
* **A fallback must not eat the original error.** That row's captured reason was the
  *retry's* message, which named neither the argument nor the cap. The retry now re-raises
  the first exception.
* **`TSFMTRL_DEBUG=1`** prints the traceback of anything the matrix swallows, from the
  row capture and from every rung of the reward-mode ladder. The report keeps only
  messages (one bad model must not end a 46-model run), which is right for the table and
  useless for fixing a failure that only appears inside the matrix. It goes to stderr;
  stdout stays the data channel.
* **Vendor loops log on their own interval.** A Chronos-2/Kronos vendor fine-tune at smoke
  size produces no loss pair in the matrix cell; recorded rather than hidden
  (`docs/03_fine_tune.md`).
## Normalising an interpreter path can leave the venv

Recording *which* interpreter produced a row means storing its path, and the obvious
`Path(py).resolve()` is a trap: `env/bin/python` is a symlink to the base interpreter that
built the venv, so `resolve()` hands back something under uv's python directory with none
of the family's pinned packages. The run then dies on `import numpy` in 0.0 s and the
failure looks like a broken model environment. `scripts/sweep_evaluate.py` uses
`os.path.abspath()` (lexical, no symlink following) and `tests/test_sweep_paths.py` keeps
that true — the same distinction that makes `sys.executable` and `realpath(sys.executable)`
different tools.

## A family env can silently lose its GPU

Lock-file rebuilds on aarch64 resolved CPU wheels in three of the twelve DGX environments
(`toto` → `torch 2.7.0+cpu`, `timesfm20` and `new` → `torch 2.8.0+cpu`). Nothing about the
run looked wrong: `--device auto` quietly chose CPU, the numbers came out plausible, and a
run that *asked* for CUDA failed with `Torch not compiled with CUDA enabled` — which reads
like a model or harness failure and is neither. Two mitigations are now in the tooling:

* the sweep probes each env's interpreter once (`torch.__version__`,
  `torch.cuda.is_available()`, device name), prints a NOTE when `--device cuda` was asked
  for and the env has no CUDA, and writes the probe into every record and into the report's
  provenance table;
* the reward seeding script does the same for the child interpreter before spending
  minutes on a doomed run.

The lesson generalises: **a benchmark row is its environment.** A CPU row and a GPU row for
the same model are different experiments, and "it ran" is not evidence about which one you
got.

## Seventh pass: 101 series, and three ways a number can be wrong without any code erroring

### A checkpoint's context length is an interface, not a suggestion

`granite-timeseries-ttm-r3` declares `context_length = 512`. Fed the protocol window of 256
it pads its own input with zeros, and the whole forecast comes back scaled by the fraction
of real history: measured on one equity series, the mean prediction is 0.25x the context at
ctx 128, **0.50x at ctx 256**, 0.99x at 512. In the first pass-3 ranking that read as
"median MASE 13x naive, 0% of series beaten" for a model whose weights are fine. The
artifact was our window.

Fix: `TSFMModel.min_context` (read by the Granite adapter from the checkpoint config).
`forecast_eval` raises the benchmark context to it when the span allows and writes
`context raised 256->512 (checkpoint contract)` into the row's `ctx_note`; when the span
cannot supply it the adapter pads on the left with the oldest observation instead of letting
vendor code feed zeros, and says so in `Pred.meta`. The row's `ctx` column then shows 512
for those models, which is why the effectiveness table has a `ctx` column per row at all.

The honest residual: with the contract honoured, TTM r2 behaves normally (MASE ~3-9) and
**r3 still loses badly on financial series** because its forecast is nearly flat (std ~0.8
against a truth std ~8 on a volatile equity) - an approximate mean-forecaster on
low-volatility price data. Finding and artifact are now separable; before, they were the
same number.

`min_context` is set only where the padding artifact was measured. A contract probe over the
sweep models (one equity series, level ratio at ctx 256 versus the declared length) found
FlowState declaring 2048 and PatchTST-FM 8192 while both score normally at 256 - FlowState
actually gets *worse* at its declared length (level ratio 0.587 at 2048 against 1.135 at
256). Those models accept short windows honestly, so raising their context would change the
experiment rather than repair it. The rule is: measure the padding behavior, then declare
the contract - do not read a config number and treat it as a requirement.

### NaN is a failed measurement, not a metric value

TTM r3 also emits an all-NaN point head *intermittently*: the same 512-point window was
clean in one process and NaN in the next. Two wrongs were hiding in the old path: the
vendor model returned NaN without raising, and `forecast_report` is nan-safe, so it scored
the surviving 3 of 4 windows while the row still claimed `n_windows: 4`. My own ranking
script then aggregated a column containing NaN and printed "median MASE nan" for the
model - which looks like a joke and reads like one.

Now: retry each affected window alone (a batch can poison a model's own per-instance
normalisation), drop whatever is still non-finite from the scoring set, record
`pred_nan_windows` plus an `error` string, set `n_windows` to the number actually scored,
and emit `None` rather than NaN in metric columns.

### Report sections that iterate the wrong collection under-count across lanes

`report()` merges cells per `(model, series, span, horizon)` but keeps one *raw record* per
model in `rec_by_model`. The tables read the merged cells; the "Not scoreable on some
series" section and the new untrained-head section read `recs`. With two GPU lanes each
carrying half the corpus, the flag list showed 33 of 66 flagged MOMENT series, and the
covariate-panel section was under-counted in pass 2 the same way. Both now read the merged
rows, and a model that scored anywhere is no longer listed as dataset-unscoreable.
Symptom to remember: a summary section whose count is exactly one lane's worth.

### Shared cards: free at launch is not free during

`chronos-t5-base` lost 33 of 66 series to `torch.OutOfMemoryError` with the error reporting
44.94 MiB free, minutes after `nvidia-smi` showed ~10 GiB - another user's process grew to
26 GiB mid-run, and our own peak was 7.09 GiB. The lane runner's `--min-free-mib` guard is
necessary, not sufficient. Two operational rules follow: pin `CUDA_VISIBLE_DEVICES` to the
index you chose (a retry that picked "cuda" from a free-memory total landed on the
contended GPU 0 anyway), and treat a mid-run OOM as an environment event to re-run, never
as a model result. Where the cards stay full, the retry runs on CPU lanes and the rows
carry `device: cpu`.

### Two small tooling facts from this pass

- `--md path/to/report.md` now creates the parent directory; before, the first invocation
  into a fresh run directory died on `FileNotFoundError` after doing all the merge work.
- The trux `core` env has no ruff, so `scripts/check.sh` stops at the lint stage there.
  Tests, registry consistency and the mini matrix are run on trux; lint stays a laptop
  gate. That is a real gap in the multi-machine story, not a covered one.

### Syncing a checkout with `rsync --delete` deletes the environments living inside it

The harness exists twice on the laptop: the standalone repo and `trl/` in this checkout,
which is what the other machines fetch. Bringing the standalone repo up to date with
`rsync -a --delete` overwrote more than the code. Two things were gone before I looked at
the file list:

- **`.github/workflows/docs.yml`** — this tree has no `.github/`, so `--delete` removed the
  Pages build. Nothing in the harness test suite notices a missing CI file.
- **Every family venv under `envs/`** (several GB, built over days) — same reason: they are
  in the destination and not in the source. The locks and `envs.toml` survived, so the fix
  is `scripts/make_envs.sh`, not an archaeology hunt, but the run I was mid-way through on
  the laptop was over.

Nothing tracked by git was lost (`git status` showed 89 modified/added paths, 0 deletions),
which is the only reason this was an afternoon and not a data-loss incident. The rule now:
when one checkout feeds another, list what may move (`git ls-files`, plus the probe and
report text manifests) and copy *that*, or keep `--delete` off and let git show what is
missing. A working tree that holds installed environments is never a safe rsync target.

Because the two checkouts drift, the standalone repo is now pushed (`master` @ the sync
commit, plus a `pre-sync-2026-11-26` branch holding the state before it), and this tree
stays the source the machines clone. Docs edits are meant to land in both; where they
haven't yet, this file is the one to trust.

### A row's `model` field is the adapter's family key, and ten checkpoints collide on it

The registry key (`chronos-bolt-small`) is not what lands in a result row. The adapter
writes its *family* key (`chronos-bolt`), so one row-level id can come from several
checkpoints. Across the registry there are ten such collisions:

```
chronos-bolt   <- chronos-bolt-small, chronos-bolt-base
chronos-t5     <- chronos-t5-small,   chronos-t5-base
ttm-r2         <- ttm-r2,             ttm-r3
kronos-small   <- kronos-small,       kronos-base,  kronos-mini
moment-1-large <- moment-1-small,     moment-1-large
moirai-1.1, toto-2-22m, kairos-10m, ... (same pattern)
```

The report was safe (it keys merges on the record-level registry key and stamps
`model_key` when reading), but nothing said that out loud, and my own comparison script
keyed on the row field. The result was a fabricated finding: "`chronos-bolt` disagrees by
3-40% between the pass-3 run and today's re-run" — which was really *bolt-small vs
bolt-base* being averaged together, and it nearly sent me chasing cuBLAS nondeterminism
across GPU generations. After the fix, GB10 (gpu7) and RTX 6000 Ada agree **bit-exactly**
(0.000%) on every overlapping cell, baselines and neural models alike.

Two changes: every written row now carries `model_key` (`stamp_rows`), and the merge keeps
its `model_key` stamp on read for older files. `probes/trux/cross_machine_check.py` keys
on the registry key for the same reason. The trap generalises: a label chosen by the code
under test is not an identifier for the analysis on top of it.

**Follow-up, because the first reading of the duplicates was also wrong.** With the key
fixed, the merge's 404 repeated cell keys were re-scored by protocol signature (ctx,
ctx_note, device, code_sha, torch). They are overwhelmingly *supersessions*: a TTM cell
measured at ctx 256 (MASE 30.1, the zero-padding artifact) and the same cell at its
contract ctx 512 (MASE 4.3). Only repeats with an identical signature can be read as run
variance, and once separated they are small: 254 repeated comparisons on trux - pass-3
lanes at sha `1d2f3ae` against re-runs at `28fd7e7` - agree to 0.143%, and the DGX Spark
matches trux to 0.000%. So the journal's earlier "neural cells vary 2-200% run to run" was
one artifact (the collided key) compounding another (a fixed protocol change counted as
noise). Reproducibility in this benchmark is now measured, not assumed, and the report
section says which of the two a repeated key is before it quotes a number.

### A GPU that answers `[N/A]` for memory is a scheduler bug waiting to happen

The DGX Spark (GB10, aarch64, unified memory) reports `memory.free` and `memory.total`
as `[N/A]` while `torch.cuda.is_available()` is true and the device works. The lane
scheduler parsed those rows away, so `gpu_free()` returned an empty pool and *every* lane
fell back to CPU on a machine with a working GPU — the row would still have been written,
just labelled `device: cpu`, which is how a new arm of the benchmark would have produced a
table of CPU numbers under a GPU heading. `parse_gpu_rows` now drops only non-numeric
memory rows and tolerates extra trailing columns; `gpu_free()` falls back to
`torch.cuda.mem_get_info()` and tags the row `reported_by: torch`, because a number used
for scheduling should say where it came from. Two facts to keep straight: `nvidia-smi`
knowing the GPU exists is not the same as knowing its free memory, and strict unpacking of
a tool's csv turns a column change into silently empty output.

## Sixth pass: moving the harness onto other machines

The same code went to trux (2× RTX 6000 Ada) and watson (SLURM, A6000). What broke was not
the models; it was the boundary between "a machine that has this project" and "a machine
that has the right *environments*".

**A recorded skip is sticky.** On watson the first run had no `envs/local.json`, so all 16
models were written as env-skips — the resolver refusing to run a family in a foreign
interpreter, exactly as designed. After writing the mapping, the next run printed
`0 ok / 0 failed / 0 env-skipped`, which reads like a working sweep: the models already
appear in the incremental JSONL, so resume treats them as done. Fixing the environment is
not enough — the log needs `--fresh`. The merge rule ("rows beat errors") hides the
symptom in a report, not the trap at launch.

**Seven of twelve lock files do not re-resolve** (`core`, `moment`, `new`, `timemoe`,
`tirex2`, `toto`, `ttm`): `uv` proves the frozen set contradicts itself, e.g. `gluonts`
wants `pandas<2.2` while the freeze pins `2.3.3`. Those freezes were taken from layered
installs, so they are a *record* of what worked, not a recipe a solver can reproduce.
`uv pip install --no-deps -r <lock>` reinstalls exactly the recorded set; a solver-driven
install does not. `core.lock.txt` also carries no `torch` pin by design (torch is
platform-specific and installed separately), which surprises anyone expecting one command
to rebuild an env. Related drift: `envs.toml` notes `transformers==4.57.6` for core while
the committed lock pins `5.22.1` — the lock records a *later*, drifted env, not the one
that produced the published matrix.

**A plausible command line is not the real one.** The lane runner had been written against
a remembered CLI: `--windows` and positional series do not exist (`--max-windows`,
repeatable `--series`). It died at launch, which is the cheap way to find out. Now
`probes/trux/selftest/` keeps a stub whose argparse mirrors `sweep_evaluate.py --help`, so
an interface drift fails a 1-second test instead of a 12-hour sweep. Note that the real CLI
inherits argparse prefix matching, so a typo like `--windows=3` is *silently accepted* as
`--max-windows=3`; the stub forbids abbreviations so a runner cannot lean on that hazard.

**Flattening the corpus needs a rule, not a shrug.** The corpus stores parquet splits, the
sweep documents `timestamp,close` CSVs. Flattening found 19 series whose splits share
timestamps — always *across* splits, never within one, which is lookback context reuse.
But `open/sp500_daily`'s train range runs *past* its validation range, with validation
entirely inside train. Zero-shot rows sliding over the whole series are unaffected;
calling those splits held-out data is wrong. The converter now records overlap kind and
range per series, and refuses to flatten a timestamp that carries *different* values.

**A green cell can still be an untrained head.** MOMENT's released checkpoint contains one
head — `head.linear` (8, d_model), reconstruction. Asking for `task_name="forecasting"`
loads that head unchanged (transformers reports "0 missing keys" because the new head has
different *shapes*), and the `init()` the model card tells you to call replaces it with a
randomly initialised one. The compatibility matrix therefore shows a pass; the benchmark
rows say what ran — MASE 7.0–19.6 where a naive forecast gets 2.1–8.4, `large` within ~1 of
`small`. Rows now carry `head_from_checkpoint` and `head_trained_here`, and the report
lists flagged models under "Flagged: the scored head was never trained" while keeping the
numbers.

## Fine-tune pass: the lr moved the number, and the job script lost its own summary

MOMENT is now the only family with both zero-shot and fine-tuned cells on the same series
(12 series, A6000, `reports/moment_ft_watson/lane_ft.jsonl`). Two findings:

**Learning rate dominated the adaptation scope.** At lr 1e-6, training only the head left
the score where zero-shot left it (2.42x naive vs 2.56x); at lr 1e-4 with only the head
trainable it reached 1.67x, matching a full fine-tune of the whole model. Sweeping
`heads_only` vs `full` while leaving lr at a default compares two things when one knob
matters. `--configs` therefore accepts `peft:budget[:lr]` and the sweep records `lr` per
run, so a null result can be attributed instead of being read as a property of the family.

**A batch job can succeed and still print nothing useful.** The MOMENT job finished —
2373 s, 156 run records, `done in 2373s -> reports/moment_ft_watson/lane_ft.jsonl` — and
then its trailing summary line failed with `envs/core/bin/python: No such file or
directory`, because the script used a path relative to a directory SLURM had not made the
cwd. The results survived only because the lane file is written as it goes; a harness that
emits its numbers at the end of the script would have produced a successful-looking job
with no data. Job scripts should use absolute paths for their own environment, and the
lane file should be append-as-you-go (it is).

## A dispatched bridge job is one file: three failures, one rule

Getting the GB10 replication job to run took four attempts, and all three failures were the
same fact approached from a different angle: `trux_dispatch.sh` transfers **one** file to the
worker's inbox, and the worker runs it in a fresh shell on the far machine.

| attempt | what was assumed | what happened |
|---|---|---|
| job_006 | `SERIES_LIST=... bash trux_dispatch.sh job.sh` would carry the argument | `FATAL: SERIES_LIST required` - the env never crosses |
| job_007 | the job may read a sibling file (`gpu7_watson12.series`) "next to the script" | `cat: .../inbox/gpu7_watson12.series: No such file` - only the job file lands there |
| job_008 | the job may `exec` the real runner from the repo clone by relative path | `bash: .../inbox/gpu7_030_replicate.sh: No such file` - `$0` is in the inbox, not the repo |

The job that works (job_009+) is generated: the runner body plus the series list inlined.
Two side lessons from the same loop. The generated file had a literal `\n` spliced into a
line continuation, which bash read as the word `n` passed to `git pull`, so the repo never
advanced and the failure surfaced as a `WARN` - a generator that concatenates strings needs
the same lint as hand-written code. And `--models chronos2` died with `unknown model(s)`:
the registry *key* is `chronos-2` while the *family* is `chronos2`; the sweep validating
model names at startup, rather than defaulting to a substring match, is what made that a
five-second failure instead of a silently empty lane.

## The same series name can be two different measurements: CSV vs arena directory

The GB10 replication lane was the first real overlap between machines: 12 series watson had
already scored, 216 shared cells. `probes/trux/cross_machine_check.py` called 212 of them
disagreements, with MASE off by 14.98%, 37.68%, 30.80% — up to 122.9%. None of it was
hardware. The signature of the bug was in what *agreed*: RMSE matched to about the eighth
decimal, directional accuracy was identical, and both horizons of one series were off by
exactly the same ratio.

`mase(y, p, train)` is `mean|y-p| / mean|diff(train)|`. The numerator is the forecast; the
denominator is a property of **which split the loader invented**. trux and gpu7 read the
arena directory `std_equity_1d_10104` (publisher's train/validation/test parquets, train
n=1510, naive MAE 0.375671). Watson's pass-3 job read `~/pass3_csv/std_equity_1d_10104.csv`,
a flat file, so `SeriesDataset` synthesized a 70/15/15 split (train n=1937, naive MAE
0.431936). Their ratio is 0.8697; its reciprocal 1.1498 is the observed MASE ratio for that
series to four decimals. The forecasts were the same numbers.

What that does to the published table:

* **Survives**: `beats naive` per cell (the naive baseline in the same lane shares the
  denominator), and the unscaled metrics — `rmse`, `mae`, `dir_acc`. Cross-machine on those:
  bit-exact.
* **Does not survive**: any absolute MASE comparison between the watson block (35 series,
  CSV) and the trux block (66 series, arena dirs). They are two scales of the same quantity,
  differing per series by 1.0x–2.2x depending on where the publisher's train boundary sits
  relative to 70%.

The fix is a fingerprint, not a rule to remember. `SeriesDataset` now stamps
`data_hash` — sha1 over the loaded values *and* the split boundaries, 12 hex chars — and
`data_kind` (`arena_dir` / `csv` / `parquet_file` / `in-memory` / `frame`) into `meta`, and
`forecast_eval` copies both onto every scored row. `scripts/sweep_evaluate.py` adds the lane
meta field `data_kinds`, and both consumers use it: the merge's protocol signature includes
the data path, so a CSV/arena repeat of one cell is filed under *superseded* rather than
*run variance*; `cross_machine_check.py` refuses to call it a machine disagreement and exits
2 (inconclusive), with a fallback for legacy lanes that predate the row-level stamp — their
meta line still lists the input names, and `.csv` versus a bare directory name is enough to
tell which split was loaded.

The general form: **a series name identifies a file, not a measurement.** Anything scaled by
a quantity derived from the data (MASE by the train-split naive error, any z-scored metric,
any "beat baseline by X%" claim) silently changes when the data path changes, and the only
durable defence is to write down the bytes and the split next to the number.

## `--device` is a record, not a constraint - until the adapter enforces it

A trux lane run with `--device cpu` wrote `device: cpu` into its meta and then died with
`CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 47.37 GiB of
which 14.94 MiB is free` - from TimesFM-2.5, which was never supposed to touch GPU 0. In the
vendor package (`timesfm/timesfm_2p5/timesfm_2p5_torch.py:72`) `TimesFMModel.__init__` sets
`self.device = torch.device("cuda:0")` whenever `torch.cuda.is_available()` and calls
`self.to(self.device)` **inside the constructor**; `from_pretrained` accepts no device
argument, and `compile()` later sizes work as `per_core_batch_size * model.device_count`.

Two rounds of pinning failed before one worked. `.to("cpu")` after `compile()` was too late
and left the cached `device`/`device_count` pointing at the GPU; pinning right after
`from_pretrained` still OOMed because the *load itself* allocates there. What works
(`TimesFM25._load_off_gpu`): when cpu was requested, hide the device for the duration of
construction (`torch.cuda.is_available -> False`, `device_count -> 0`), then move the module,
correct the cached attributes, and verify placement - raising `UnavailableError` rather than
writing a row that names a device it did not run on. Verified on the saturated trux card: one
series, one window, ok in 4.6 s, CPU.

The rule for every adapter: **the device column is a claim you have to earn.** Check
`next(model.parameters()).device` after loading and refuse on mismatch. An adapter that
ignores `device=` does not just run in the wrong place - it falsifies the provenance of the
row, which is the one thing a benchmark row is supposed to be able to do.

## One model family is device-sensitive; the rest agree bit-for-bit across machines

With the data path finally held constant (both lanes reading arena directories), the GB10
replication lane and a trux lane give 216 shared cells (12 series x 2 horizons x 9 models,
once the TimesFM CPU lane was re-run), gpu7 on CUDA vs trux on CPU:

| model_key | n cells | median rel. diff (mase) | max |
|---|--:|--:|--:|
| naive, seasonal_naive, arima, chronos-2, chronos-bolt-small, chronos-bolt-base, timesfm-2.5-200m, ttm-r2 | 24 each | 0.000% | 0.000% |
| chronos-t5-small | 24 | 3.48% | 13.44% |

Eight of nine families are identical to the last decimal across an aarch64/CUDA box and an
x86/CPU box - which is a stronger reproducibility statement than anything this project had
before. The exception is instructive: Chronos-T5 is the one path that *generates* (beam/
sample decoding), and generation is where a different BLAS, a different kernel, or a bf16
accumulation compounds into different sampled quantiles. Its pass-3 rows already mix CUDA
(33 cells) and CPU (33 cells, the OOM retries), so those cells are not the same measurement -
read `device` before comparing TTM-style, and treat T5 cross-device gaps of a few percent as
expected rather than as a bug.

Three machines, same protocol, same bytes (2026-11-27). Watson's 35-series lane was re-run
reading arena directories (`reports/pass4_watson_arena`: RTX A6000, CUDA, 16 models, 0 failures), so its 12 overlap series now share a data path with gpu7 and trux. Pairwise MASE on those cells:

| machine pair | hardware | identical | out of tolerance |
|---|---|---|---|
| gpu7 \| sedan-gpu1 | GB10 CUDA \| A6000 CUDA | 8 of 9 families at 0.000% | 18 cells, all `chronos-t5-small` (median 5.38%, max 11.32%) |
| gpu7 \| trux-tsubame | GB10 CUDA \| x86 CPU | 8 of 9 families at 0.000% | T5 only |
| sedan-gpu1 \| trux-tsubame | A6000 CUDA \| x86 CPU | 8 of 9 families at 0.000% | T5 only |

Two CUDA stacks that share nothing - different GPU architecture, different torch build - reproduce each other exactly. That is the strongest form of the claim available, and it locates the exception in the *decoding path*: `chronos-t5-base` ran only on watson in this pass, so it still has no cross-machine pair.

**The 66-series replication (pass 5, 2026-11-27) settles it.** Watson re-scored gpu7's 66 series
on an A6000 (`reports/pass5_watson_66`, node `sedan-gpu3`), giving **1,452 shared cells** —
66 series x 2 horizons x 11 models — with `gpu7` (GB10, CUDA, torch 2.9.1+cu130) and watson
(A6000, CUDA, torch 2.7.1+cu128) reading the same arena bytes:

| model_key | cells | median diff | max diff | cells > 2% |
|---|--:|--:|--:|--:|
| naive, seasonal_naive, arima, theta, moving_average, chronos-2, chronos-bolt-small, chronos-bolt-base, timesfm-2.5-200m, ttm-r2 | 132 each | 0.000% | 0.000% | 0 |
| chronos-t5-small | 132 | 5.730% | 17.406% | 109 |

Ten of eleven families are bit-identical to the last decimal on 132 cells each; every out-of-tolerance
cell in the whole 1,452-cell stratum belongs to `chronos-t5-small` (rmse behaves the same way: median
0.000% everywhere except T5, where it reaches 23.2%). The same generator produced `ttm-r2` bit-exactly,
and TTM had been suspected of device sensitivity earlier — it was the context-length contract, not the
device. So the rule to carry forward is narrow and checkable: **anything that samples tokens at decode
time may differ across stacks by several percent; everything else should be compared to zero.**


## A 2 MB documentation page is published but not served

`scripts/build_docs.sh` builds strictly and CI commits the site to `gh-pages`. The benchmark
page used to be the full per-cell table (3,628 rows -> 2.19 MB of HTML). GitHub Pages returned
404 for `/effectiveness/` while `/matrix/` and every other page served, and the file itself
was reachable at `/effectiveness/index.html` - a URL nothing links to, because mkdocs uses
directory URLs. The page was invisible exactly where the navigation pointed.

`--summary-only` now generates aggregates for the site (headline ranking, provenance,
interpreter table); the full per-cell table stays in the repo's merged report, which is also
the readable choice.

Separately, `actions/deploy-pages` never worked on this account: every attempt failed with
"Failed to create deployment (status: 404) ... Cannot find any run with github.run_id", both
before and after the repository went public and after recreating Pages with
`build_type=workflow`. The branch-push route (`git push HEAD:gh-pages`) does publish - the
same mechanism the TSFM4Finance leaderboard site uses - so the docs workflow does that now.

## Re-dispatching a bridge job under a used number never runs

The Spark/GPU7 worker picks the first `inbox/*.sh` that has **no** matching
`outbox/<name>.log`, because trux leaves the job file in the inbox until it is reaped and
`rsync` would otherwise hand the same job back forever. That rule makes job numbers
single-use: after `job_013` ran and reported, dropping a corrected `job_013` back into the
inbox left it sitting there for an hour, with a live worker and a fresh heartbeat, and no
error anywhere — from the worker's point of view it was already finished.

Two facts to keep separate: the run I was waiting for had actually completed (`job_013.log`
on trux: 11 models ok / 0 failed over 66 series), and the file I re-dispatched was a stale
duplicate of my own making. Housekeeping rule: a dispatch is a new number or nothing; if a
name has a log, the answer is already in that log.

## A hand-written file on a remote clone blocks the pull that brings the next job

Both remote harness clones stopped fast-forwarding for the same reason, an hour apart. On trux:
`reports/pass4_watson_arena/lane_arena.jsonl`, left there as scratch output in an earlier session and
committed from the laptop afterwards. On watson: `probes/watson/watson_pass5_66.sh`, written by hand
on the remote box and committed from the laptop afterwards. Git refuses to overwrite an untracked
file during a merge, so `git pull --ff-only` aborts with "The following untracked working tree files
would be overwritten" - and a dispatch script that pipes its own output through `tail -4` shows the
pull line but not the abort, so the job runs on the *old* code and records an old `code_sha`.

The fix is mechanical (move the file aside, pull, compare, delete if identical), and in both cases
the scratch copy was byte-identical to the committed one. The lesson is about direction of travel:
artifacts written on a compute box have to come back to the control machine and into git, or they
turn the compute box into an island whose pull is permanently broken. Anything worth keeping on a
remote lane belongs in the repo; anything not worth keeping belongs in a directory git does not
track (`/scratch/<job>/`).

## `--max-windows` draws the test windows at random, so a different seed is a different measurement

Reproducibility work had ruled out hardware, code drift and data path. What it had not measured
was the sampling design. `windows()` strides the test span and, when `max_windows` caps the
count, picks the subset **uniformly at random** (same seed, same windows — which is exactly why
the fleet never noticed: every model on every machine used seed 0 and therefore the *same*
windows, so cross-machine comparisons stayed bit-exact).

What seed 0 vs seed 1 means on four arena series (x86 CPU, identical bytes, identical code):

| windows/cell | naive's own MASE, seed-to-seed | chronos-bolt MASE | paired ratio model/naive |
|---:|---|---|---|
| 8 (fleet default) | median **55%**, max 100% | median 54%, max 124% | median **12%**, max 25% |
| 32 | median 21%, max 57% | median 13%, max 69% | median **7%**, max 17% |

Read the first column again: the *naive baseline's own MASE* doubles between seeds. A per-cell
MASE at 8 windows is a one-digit estimate, not a measurement.

What survives: paired comparisons inside one seed (every model sees the same windows), and
aggregate shares over hundreds of cells — the leaderboard's "41% vs 40%" is a count across
~1,400 cells, so its noise is far smaller than any single cell's. What does not survive: calling
one cell a win or a loss when its ratio sits within ~12% of parity, and any per-cell number
quoted without its window count.

Two fixes, both about not letting the merge machinery blur the sample:

* `proto()` — the merge's protocol signature — now carries `n_windows` and the lane `seed`, so a
  32-window row cannot silently supersede an 8-window row; it becomes a supersession whose gap
  measures the protocol change.
* `cross_machine_check.py` refuses to compare rows whose `n_windows` differ, as it already
  refuses rows whose `data_hash` differ.

The ranking table prints `win` per model and states the paired-ratio noise next to itself. For
any pass whose per-cell numbers will be quoted individually, raise `--max-windows` to 32+ and
budget roughly double the wall time.

## A partitioned corpus makes a merged report look cross-checked when it is not

The multi-machine fleet was designed to avoid duplicate work: watson scored 35 series,
trux/gpu7 scored 66, and the two sets **do not intersect**. A merged 101-series report is
therefore not 101 series measured twice; it is 101 series measured once each. When the
pass-4 arena lanes were fed to the cross-machine checker it reported
`cells seen: 2546, on >=2 machines: 216` and zero gpu7-vs-watson pairs — with no disagreement
flag, because there was nothing to compare.

This is the mirror image of the duplicate-key trap. Too many rows for one cell hides which
measurement applies; too few rows per cell silently removes the ability to check at all.
`probes/watson/watson_pass5_66.sh` re-scores gpu7's 66 series on the A6000 (different GPU
architecture, different torch build, same arena parquet bytes) so the bit-exactness claim
that currently rests on 12 series rests on 66.

**Resolved the same day (pass 5).** Watson re-scored gpu7's 66 series (`reports/pass5_watson_66`,
SLURM 3291, 16 models, 0 failures). The cross-machine stratum went from 216 cells to **1,452**, and
the merged arena report now contains real replication for 66 of 101 series rather than a partition
dressed up as coverage. The lesson generalises: a fleet that avoids duplicate work is efficient and
unfalsifiable at the same time — decide *before* dispatching which cells will be measured twice.


## "Superseded by a fix" and "re-measured on another machine" are not the same sentence

The merge report classifies repeated `(model, series, span, horizon)` keys by their protocol
signature, and until now every differing pair was described as *superseded by a fix*, with the gap
called "the size of that fix". That was true of pass 3 (ctx 256 vs 512, CSV vs arena) and false of
pass 4/5, where **1,668** pairs differ only in `device`, `torch` build and `code_sha`: same context,
same `data_hash`, same window count. Calling those a correction tells a reader the older numbers were
wrong; they are the reproducibility measurement, and their gap (median 0.000%, max 17.4%, all of it
`chronos-t5-small`) is the answer to "how much does the hardware cost".

`proto()` is now `proto_core()` (what was measured: ctx, ctx_note, data_kind, data_hash, n_windows)
plus the stack fields, and the report prints three buckets that mean different things: **re-measured
on another stack** (gap = reproducibility), **superseded by a protocol change** (gap = the size of
that change), **identical signature** (gap = run variance). Two tests pin the direction: the same
bytes on two machines must not say "protocol change", and 8-vs-32 windows on one machine must not say
"another stack".
