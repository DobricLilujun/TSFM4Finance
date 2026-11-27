# 3 · Fine-tuning

> **Why this exists in a benchmark harness.** Fine-tuning is not the deliverable and the
> matrix's fine-tune column is not a result. It is a **capability gate**: for each
> architecture we prove there is a differentiable objective — the vendor's own where one
> exists — that gradients flow through, that a trainable subset can be selected, and that
> the run is checkpointable with provenance. That gate is what makes adapted rows in a
> benchmark table auditable (which base, which PEFT level, which parameters, which split
> boundary). Accuracy comparisons live in `docs/09_benchmark.md`; nothing in this file or
> in the matrix claims fine-tuning improves out-of-sample skill.

```python
from tsfm_trl import ForecastTrainer, ModelHub, TrainingConfig
from tsfm_trl.data.fixtures import tiny_finance

m   = ModelHub.load("moirai-2.0-R-small", device="cuda")
cfg = TrainingConfig(budget="fs16", peft="lora", ctx=256, horizon=16, seed=0)
res = ForecastTrainer(m, cfg).fit(tiny_finance())

res.status, res.steps, res.loss_first, res.loss_last   # 'smoke', 8, 3.027, 3.111
res.checkpoint                                          # written path
```

CLI equivalent:

```bash
tsfm-trl fit --model moirai-2.0-R-small --series px.csv --peft lora --budget fs16 \
             --ctx 256 --horizon 16 --ckpt reports/ft/px
```

## Data

`fit` accepts any of:

* a `SeriesDataset` (or a CSV/Parquet path, or a DataFrame) — windows are built for you
  with a train/val/test split,
* `{"windows": [(ctx, y), ...], "info": {...}}` — pre-cut windows, what the matrix uses,
* `{"dataset": ds}` — explicit.

Windows come from `data.series.windows(values, horizon, ctx, stride=…)`; the split
index is stored in the checkpoint so an evaluation can never accidentally train on the
test span.

## How much to train: budgets, not step counts

`TrainingConfig(budget=...)` picks a size, and the trainer keeps the *shape* of the run
stable across families:

| budget | windows | steps | lr |
|---|---|---|---|
| `fs16` (default for the matrix) | 16 | 40 (≤ that) | 1e-5 |
| `fs64` | 64 | 120 | 1e-5 |
| `full` | all | 3 epochs | 5e-6 |

`smoke=True` caps steps to 3 — that is what `doctor` uses, because the matrix checks
that a fine-tune **runs**, not that it converges. Losses are still reported
(`✅ (0.587→0.612)`), so a diverging run is visible rather than hidden.

## Three levels of invasiveness

```python
TrainingConfig(peft="heads_only")   # only the forecast head(s) train
TrainingConfig(peft="lora", lora_r=8, lora_targets=["q", "v"])
TrainingConfig(peft="full")         # everything (fine on ≤300 M at smoke size)
```

* `heads_only` uses the adapter's `head_prefixes` (e.g. `("decoder", "projection",
  "head", "p21")` for Kairos) — the names come from the vendor modules, and the list
  that actually became trainable is stored in the checkpoint.
* `lora` goes through `training.peft.apply_lora`, and it **freezes the trunk first**.
  It did not always: `_inject` froze only the `nn.Linear`s it wrapped, while every HF
  checkpoint loads with `requires_grad=True`, so `peft="lora"` had been "full fine-tune
  plus a few adapters" for most families — `t0-gamma` reported 156 M trainable
  parameters, which is its trunk, not an adapter budget. A frozen trunk plus adapters is
  what a LoRA budget means; the current counts are 1.08 M for `chronos-t5-small`,
  2.57 M for `MOMENT-1-large`, 59.6 k for `TTM-r2`. `merge_lora` folds adapters into the
  base weights for export.
* `full` is refused for families where the backbone cannot take a gradient (see below).

Loss follows the model: `loss_kind = "quantile"` families are trained with pinball on
their own head, `"token"` families (Chronos-T5, Time-MoE, Kronos) with cross-entropy on
their discrete tokens, and vendor-loop families (Kronos) delegate to the vendor's own
trainer via `vendor_fit`. Two families train on an objective that is neither:

* **Sundial** trains its **TimeFlow head** — the checkpoint repo's own `FlowLoss`
  (velocity matching over patches), fed teacher-forced hidden states so that the state
  after patch *i* supervises patch *i+1*, which is what `generate` actually consumes.
  One thing has to be said plainly: the released checkpoint ships
  `output_token_lens=[720]`, a *generation* setting — `forward(labels=…)` trims its
  labels to a length an input window never reaches, so the shipped 720-wide flow MLP
  cannot supervise a short window. Fine-tuning therefore rebuilds the flow head at the
  training patch length (**re-initialised, ~11.4 M params**) while the 12-layer trunk
  carries over. The trunk transfers; the head does not. That is a real limitation of
  the public checkpoint, not of the harness, and it is why `prepare_training` logs it.
* **t0 (α/β/γ)** supervises the vendor's own differentiable pass: `_single_pass` is
  `scaler.scale_input → forward → scaler.rescale_predictions`, so its output is in data
  space and carries a graph, and `forward` is documented as "predict per-patch quantiles
  for every patch position (one differentiable pass)". The alignment is t0's own —
  `_predict_step` reads `predictions[:, context_patches - 1]`, i.e. the patch *after* the
  last observed one — so the step scores the model on the steps that follow the context,
  with the future never fed. Its layers are named nothing like the usual ones (`wQKV`,
  `wO`, positional `mlp.0`/`mlp.2`), so the adapter declares `lora_targets` and
  `head_prefixes=("head","decoder")`; `T0Forecaster` is itself the `nn.Module` (there is
  no `.model` attribute, which is why this family used to report no trainable module).
* **Kairos** uses the supervised forward's own loss, which comes back as **one value per
  window**; the trainer reduces it to its mean. (Before that reduction existed, the
  trainer's own `if not torch.isfinite(loss)` took the truthiness of the vector, and the
  resulting *"Boolean value of Tensor with more than one value is ambiguous"* was filed
  in the matrix as a Kairos limitation. It never was one.)

## What "not fine-tunable" means, honestly

Some families cannot be fine-tuned through an HF-style `forward` at all. They report
`— (skipped)` with the reason, and it is a statement about the model:

* **TimesFM 2.0/2.5/3.0, TiRex, FlowState inference builds** — no public training code;
  `trainable()` returns `None`, so the fine-tune column is `—`.
* **Kronos** — the vendor loop only *prepares* the CSV unless
  `extra={"run_training": True}`; the matrix records `prepared`, not `ok`.
* **MOMENT** — trainable, but its gradient checkpointing is non-deterministic, so the
  adapter disables it (otherwise torch aborts with *"different number of tensors was
  saved"*).

## Three rows that used to be wrong in the other direction

The matrix refused Moirai 1.1, Moirai-MoE and Sundial fine-tunes for a while, and on
inspection all three refusals were ours, not the models'. They are worth keeping in the
docs because each is a trap that looks like a model limitation:

* **Moirai 1.1 / Moirai-MoE**: reading quantiles off `forward()`'s samples gave a
  gradient-free "fine-tune", so the honest answer at the time was *no differentiable
  head*. The head *is* differentiable — it returns a distribution, and sampling is what
  `predict` does with it afterwards. The supervised step is uni2ts' own packed loss:
  `MoiraiForecast._convert(patch_size, past…, future_target=truth…)` → module →
  `PackedNLLLoss` over the prediction-masked patches, i.e. the objective the checkpoint
  was pretrained on. No dataset pipeline, no Lightning trainer, ~20 lines.
  One caution: `MoiraiForecast.per_sample_loss_func` is `SampleNLLLoss` (an evaluation
  loss); `MoiraiFinetune`'s default is `PackedNLLLoss`. Both came back with the same
  number in testing, so don't assume the cheap one is a sampling estimator — check which
  tensor in the graph is stochastic before blaming the estimator for a noisy curve.
* **Sundial**: the first attempt called `forward(labels=…)` with the checkpoint's own
  config and correctly reported "needs the vendor's training-script packing". What the
  vendor's fine-tune config actually does is *choose the output token length for the
  run*. Do that (rebuild `flow_loss.FlowLoss` at `input_token_len`) and the objective is
  theirs, the shapes are ours.
* **Kairos**: see above — a scalar-reduction bug in our own trainer, mistaken for a
  gradient-unsafe vendor forward.

The lesson the ladder and the reason strings exist for: a fine-tune refusal must name
the missing piece ("no target kwarg in `forward`", "labels trimmed to `output_token_lens`")
so it can be checked rather than believed.

## The ladder: memory *and* numerics

The fine-tune ladder walks `keep → bf16 → fp16 → fp32 → lr×0.1 → ctx/2 → bs=1 → CPU`.
The first rungs are memory; the `fp32` and `lr×0.1` rungs are numerical, because a
non-finite loss on step 1 is a property of that profile, not of the family (Toto-2.0-313m
trains at a tenth of the fine-tuning rate). A run that fell off the ladder says which
profile it actually trained on.

## Memory ladder

`ForecastTrainer.fit` (and `RewardTrainer.fit`) retry on OOM with progressively
cheaper profiles, and record which one was used:

```
(bs, amp, peft) = (given) → (1, bf16, given) → (1, bf16, heads_only)
                          → (1, bf16, lora)  → (1, fp32, heads_only)
```

CUDA *toolchain* failures (nvrtc, arch mismatch) are treated the same way — a CPU retry
is an environment fallback, not a silent downgrade; the checkpoint meta says
`fallbacks: [...]`.

## Checkpoints and provenance

```python
from tsfm_trl import load_finetuned
from tsfm_trl.training.checkpoints import describe

describe("reports/ft/moirai-2.0-R-small__tiny_fin.pt")
m2 = ModelHub.load("moirai-2.0-R-small")
load_finetuned(m2, "reports/ft/moirai-2.0-R-small__tiny_fin.pt")
```

A checkpoint is a state dict plus the provenance needed to refuse a wrong pairing:
base registry id, `trainable` parameter list, config, split index, loss curve, budget,
smoke flag, and any fallbacks. `load_finetuned` raises if the target model's base id
differs — a Chronos head loaded onto a Moirai is a silent nonsense machine otherwise.
`tsfm-trl ckpt <path>` prints the same record.

## Vendor training loops have no curve at smoke size

Two families fine-tune through the vendor's own loop (`loss_kind = "vendor"`): Chronos-2
(`Chronos2Pipeline.fit`, an HF `Trainer`) and Kronos (its `finetune_csv`). The adapter
attaches a `TrainerCallback` so real runs surface their loss, but a vendor loop logs on
*its* interval — at smoke size (a handful of steps) it never fires, so the matrix cell is
a bare ✅ with no `(a→b)` pair. That is a logging gap in their loop, not a missing
objective; the run's own stdout shows `train_loss`. Anything that must be compared
side-by-side should be run with `--budget fs64` or larger.
