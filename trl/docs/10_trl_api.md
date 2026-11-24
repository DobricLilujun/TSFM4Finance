# RL / reward API

The post-training layer of `tsfm-trl`: what `RewardTrainer` does, what it needs from an
adapter, and which architectures can actually be pushed with reward. Everything here is
about **capability** — "this architecture accepts this kind of post-training in this
environment" — not about whether the model got better on held-out data.

!!! tip "Read the numbers as capability"
    The reward column of the [measured matrix](matrix.md) answers *which path exists and
    runs*. Effectiveness is a separate measurement with its own
    [protocol](09_benchmark.md), and the honest demonstrators at the bottom of this page
    include negative results.

## Quick start

```python
from tsfm_trl import ModelHub
from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.doctor import _wins                    # (context, truth) pairs, train split only
from tsfm_trl.training.config import TrainingConfig
from tsfm_trl.training.reward import RewardTrainer, RewardSpec

model = ModelHub.load("chronos-t5-small", device="cuda", ctx_cap=256, batch=8)
cfg = TrainingConfig(ctx=256, horizon=21, budget="fs64", peft="lora", device="cuda")
data = {"windows": _wins(ds.split("train"), 256, 21, 64), "info": {"series": "fx"}}

res = RewardTrainer(model, cfg, n_candidates=8, kl=0.02, spec=RewardSpec()).fit(data)
print(res.status, res.extra["mode"], res.extra["reward_early_mean"],
      res.extra["reward_late_mean"], res.extra["improved"], res.extra["fallbacks"])
```

```bash
tsfm-trl reward --model chronos-t5-small --steps 48 --bs 8 --lr 2e-5 \
                --budget fs64 --peft lora --device cuda
tsfm-trl reward --model toto-2-22m --mode rft            # pin the path explicitly
tsfm-trl reward --model ttm-r2 --mode grpo               # honest failure if it can't do GRPO
```

`--mode` accepts `auto | rft | grpo | reward_descent | reward_descent_point | none`.
`auto` walks the ladder below and records every fallback; an explicit mode does not fall
back silently, so a pinned mode is how you *prove* which path an architecture has.

## The reward

Default reward (`RewardSpec`), per window, over the horizon:

```
scale   = q_hi − q_lo            # the model's own spread; falls back to |candidate| mean
point   = −|truth − candidate| / scale              scale-free accuracy
dirn    = sign(candidate − anchor) · sign(truth − anchor)
r       = clip( −w_pinball·point + w_dir·dirn , ±clip ).mean(horizon)
```

with `w_pinball=1.0`, `w_dir=0.5`, `w_cross=0.2` (crossing-quantile penalty on the numpy
path), `clip=50`. `anchor` is the last observed value of the context, so the direction
term means "did the forecast point the right way away from where the series stood" —
which is what a finance benchmark cares about, and why accuracy alone is not enough: a
flat forecast can be accurate and useless.

Scale-free by construction: rewards are comparable across series and across models,
because the denominator is the model's own uncertainty, not a per-series constant. A
custom `reward=` callable replaces it and must keep the `(B, K)` shape returned by
`reward_candidates`.

## The four paths

| mode | what the update is | requires from the adapter | who has it |
|---|---|---|---|
| `grpo` | sample K paths per window, group-relative advantage `A`, update `−A·logp` on the **sampled token sequence** | `sample_paths()` **and** `path_logprob()` | Chronos-T5 only |
| `rft` | best-of-N rejection sampling: score K candidates, supervised step toward the best one | `sample_paths()` or a sampler on the inference path | 18 models (diffusion, autoregressive-token, mixture heads) |
| `reward_descent` | gradient ascent on the reward through the head | `forward_quantiles()` (differentiable quantile head) | 5 models |
| `reward_descent_point` | same, on the point head when the quantile head lacks the needed levels | differentiable point head | 5 models |
| — | no trainable path in the public build | — | 15 declined |

`grpo` is the only path that is RL in the policy-gradient sense. `rft` is
rejection-sampling fine-tuning (the shape most TSFM papers use); `descent` is not RL at
all — it is differentiable-optimisation of a reward. The distinction is kept in the
matrix on purpose: a table that calls all of these "RL" hides which architectures are
actually policy-optimisable.

### Ladder

`mode="auto"` tries, in order of invasiveness, and records each step in
`extra["fallbacks"]`:

```
grpo → reward_descent → reward_descent_point → rft
```

plus budget fallbacks (`bs→1`, `amp`, `peft → heads_only`) when the attempt runs out of
memory. Every fallback that was taken is in the result and in the matrix row's reason —
a mode you did not ask for is a finding, not a detail. `TSFMTRL_DEBUG=1` prints the
traceback of each rejected rung instead of only its message.

## Adapter contract (what makes a model RL-able)

| method | feeds | used by |
|---|---|---|
| `trainable()` | the module to put parameters under training | all modes (`None` → the row is *declined*) |
| `training_step(contexts, truth, horizon)` | a differentiable loss | fine-tune, `rft`'s supervised step |
| `forward_quantiles(...)` | differentiable quantiles | `reward_descent` |
| `sample_paths(...)` | K candidate paths, no log-probs needed | `grpo`, `rft` |
| `path_logprob(paths)` | log-probability of a given token path | `grpo` only |

`path_logprob` is the scarce one: it exists only for discrete-token models with explicit
vocabulary logits (Chronos-T5). Diffusion/flow heads (Sundial, Kronos), mixture-of
quantiles (Moirai 1.1) and continuous regression heads (TTM, MOMENT) can sample or be
differentiated through, but have no token sequence whose probability can be written down.
HuggingFace `TRL`'s `GRPOTrainer` assumes exactly that token sequence, which is why this
layer is written against the adapter contract instead.

## Reading a result

`TrainResult` from `.fit()`:

| field | meaning |
|---|---|
| `status` | `ok` / `smoke` / `skipped` / `failed` (`skipped` = declined, `reason` explains) |
| `extra["mode"]` | the path actually used |
| `extra["reward_early_mean"]`, `["reward_late_mean"]`, `["improved"]` | **quartile means** of the reward (first quarter vs last quarter), not first step vs last step: a 96-step run can read `1.73 → 1.60` step-wise while the late quartile is clearly above the early one |
| `extra["fallbacks"]` | every budget/mode fallback taken |
| `trainable_parameters` | what was actually trainable — under `peft="lora"` the trunk is frozen first, so this is the adapter budget |
| `checkpoint` | path, empty when `smoke=True` (a 2-step checkpoint is not an artifact) |

`smoke=True` is the mode the matrix uses: two steps, no checkpoint, enough to prove the
objective is differentiable and the loop runs.

## What was measured (2026-11-21)

| model | mode | steps | trainable | reward early → late | improved |
|---|---|---:|---:|---|---|
| chronos-t5-small | `grpo` | 96 | 1,081,344 (LoRA) | 1.741 → 1.839 | yes |
| toto-2-22m | `rft` | 48 | 295,296 (LoRA) | −1.488 → −1.670 | **no** |
| toto-2-22m | `rft` | 48 | 21,915,584 (full) | −1.488 → −1.669 | **no** |
| flowstate-r1 | `reward_descent` | 96 | heads-only | −1.296 → −1.387 | **no** |

One positive and three negative results, kept visible. The negative Toto rows came from
two fixes that removed false positives: the reward's quantile-level grid, and the LoRA
trunk-freeze bug (which had been fine-tuning the whole trunk while reporting an adapter
budget). A library that only shows the improved row is not a benchmark.

## Deliberately not here

* PPO/GRPO clipping, KL to a reference policy in *logit* space (the `kl` term is an L2
  pull toward the initial weights in parameter space; `1e-3` in the class, `0.02` on the
  CLI).
* Reward models trained from preferences, or reward from a downstream portfolio P&L.
* Rollout servers (vLLM), sequence parallelism, distributed training.
* Any claim that reward post-training improves forecast quality — that is
  [09_benchmark.md](09_benchmark.md)'s job, and needs multiple seeds to say anything.
