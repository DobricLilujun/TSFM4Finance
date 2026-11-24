# 5 · Reward post-training (`RewardTrainer`)

One trainer, three mechanisms, chosen by what the model actually exposes:

| mode | what it optimises | which families can do it |
|---|---|---|
| `grpo` | group-relative policy gradient on **token log-probabilities** | token models: Chronos-T5, Time-MoE, Sundial (anything with `loss_kind="token"` + `path_logprob`) |
| `reward_descent` | direct gradient through a **differentiable quantile head** | Chronos-2/Bolt, Moirai 2, Toto, TiRex-2, FlowState |
| `reward_descent_point` | same, on a **mean-only head** (interval terms switched off) | MOMENT, small TTM releases |
| `rft` | rejection sampling / best-of-N: keep the best candidates, then supervise | any family with a `training_step` |

```python
from tsfm_trl import RewardSpec, RewardTrainer, ModelHub, TrainingConfig

m = ModelHub.load("chronos-2", device="cuda")
r = RewardTrainer(m, TrainingConfig(budget="fs16", steps=6, peft="heads_only"),
                  spec=RewardSpec(w_pinball=1.0, w_dir=0.5, w_cross=0.2),
                  n_candidates=8, kl=1e-3).fit(dataset)
r.extra["mode"]        # 'reward_descent'
```

CLI:

```bash
tsfm-trl reward --model chronos-t5 --steps 6 --candidates 8 --kl 1e-3 \
                --mode auto --budget fs16 --horizon 21 --ctx 256
```

## The default reward

`RewardSpec` is accuracy plus direction minus the pathologies that make a
probabilistic forecast useless in a portfolio:

```
reward = −w_pinball · pinball(quantiles, truth)
         + w_dir     · directional_accuracy(median vs truth)
         − w_cross   · quantile-crossing penalty
reward clipped to ±spec.clip          # outlier windows must not dominate a batch
```

## Bring your own reward

Any callable with this signature replaces it — the trading case is the reason the
signature has an `anchor`:

```python
def pnl(candidates, truth, anchor, *, cost=2e-4):
    """candidates (B,K,H), truth (B,H), anchor (B,1) -> reward (B,K)

    anchor is the last observed price: P&L is relative, so a reward that ignores it
    silently rewards forecasts that are simply close to the mean of the context.
    """
    pos = np.sign(np.median(candidates, axis=2) - anchor)          # (B,K)
    ret = np.log(truth[:, -1:] / anchor) - np.log(truth[:, -1:] - 0 * anchor)
    return pos * np.sign(ret) - cost * (pos != 0)

RewardTrainer(m, cfg, reward=pnl).fit(ds)
```

## GRPO, concretely

Per step: draw `K` candidate continuations, score them, convert rewards to
group-relative advantages `A = (r − mean)/std` across the group, then

```
loss = −Σ A · logp(candidates)  +  kl · KL(π ‖ π_ref)
```

with `π_ref` a frozen clone of the trainable parameters taken at the start of the run.
That is the ordinary GRPO update, at smoke scale (`reward_steps`, `n_candidates`),
which is what makes the matrix column meaningful: it proves the log-prob path, the
candidate sampler and the KL anchor all *run* on that family.

## Honest limits

* This is **reward post-training, not an RL benchmark**. Six steps on 16 windows is a
  capability check; `mean_reward` in the output is the mean reward of the *final* step,
  not a policy's expected return.
* `reward_descent` is not policy gradient — for a deterministic quantile head the reward
  is differentiable, and pretending otherwise would add variance for no bias-free
  benefit. The mode is recorded in the checkpoint and in `TrainResult.extra["mode"]`.
* Families with no gradient path at all (TimesFM inference builds, TiRex) report
  `— (no reward path)` rather than being run through a surrogate.
* Moirai 1.1 / Moirai-MoE are `—`: their head draws samples, so a "reward gradient"
  would be noise with a learning rate attached.

## Memory

The same ladder as fine-tuning (`bs/amp/peft` stepped down on OOM), and the same rule
that CUDA *toolchain* failures are environment problems: the run retries cheaper and
writes `fallbacks: [...]` into the checkpoint, so a table never hides the fact that a
row ran at `bs=1 bf16 heads_only`.

Runnable shape of the call: `examples/04_reward.py` (core env, one GPU, ~1 minute).

## Provenance: is this someone else's RL library?

No. `RewardTrainer` is hand-written in `tsfm_trl/training/reward.py` (~435 lines), and
the dependency graph proves it: core dependencies are
`numpy/pandas/pyarrow/scikit-learn/scipy/pyyaml/tqdm`, the `[rl]` extra is `torch` only,
and nothing in the tree imports `trl`, `trlx`, `verl`, `openrlhf` or `open_r1`. The
*algorithm* `grpo` refers to is Group Relative Policy Optimization (Shao et al. 2024, the
DeepSeek-R1 recipe); the code here is our own minimal implementation of it.

**What is faithful to GRPO**

* group-relative advantage, no value/critic network:
  `adv = (r − r.mean(axis=group)) / (r.std(axis=group) + 1e-6)`;
* K sampled continuations per window, scored by a reward that is not the loss;
* a policy-gradient update on the model's **own** log-likelihood
  (`Chronos` adapter's `path_logprob` = sum of next-token log-probs from
  `model(input_ids, labels=…)`), so the objective is the model's own language, not a
  re-derived one.

**What is deliberately simplified** — say this out loud in any paper text:

* only the **max-advantage path per group** gets the `−A·logp` term; standard GRPO
  averages the term over all K samples;
* **no PPO-style clipping and no π_old ratio** — the update is the plain REINFORCE-with-
  group-baseline form;
* the trust region is **weight-space** — a frozen clone of the trainable parameters taken
  at step 0, penalised as `kl · Σ mean((p − p_ref)²)` (`kl` defaults to 1e-3 in the class and
  0.02 on the CLI) — not the paper's
  token-level KL to π_ref;
* the reward is a forecast reward (scale-free accuracy + directional term + optional
  quantile-crossing penalty), not a program verifier.

So: *GRPO-style* is the accurate phrase; `mode="grpo"` in the matrix means
"group-relative policy gradient over the model's own sampled token paths", and it is only
reachable where an adapter exposes `sample_paths` + `path_logprob` — today the Chronos-T5
adapters (2 registered models). Kronos and Time-MoE expose `sample_paths` but no token
log-prob, so they run `rft`, which is the group-baseline limit of the same objective and
needs no likelihood.

**Why not HuggingFace TRL's `GRPOTrainer`?** It is bound to a causal-LM interface
(`logits → per-token log-probs`, a `Trainer` training loop). Most TSFM architectures are
not that: diffusion heads (Chronos-Bolt), masked-patch distribution heads (Moirai),
quantile-regression heads (TTM, FlowState, Toto), flow-matching heads (Sundial, Aurora),
RVQ tokens with a vendor training loop (Kronos). Adopting TRL would mean the RL half of
the library only works for the two Chronos variants — and it would pin a `Trainer`
dependency whose transitive requirements fight the per-family version pins that keep 11
environments working (`docs/01_install.md`).

**To make it canonical** (each step is small and localized):

1. have `path_logprob` return the per-token sequence log-prob instead of the sum, and
   average the `−A·logp` term over all K candidates;
2. snapshot π_old per iteration and add the clipped ratio;
3. replace the weight-space anchor with a token-level KL against a frozen π_ref forward.

## Demonstrators (what was actually run)

Small data (16 windows, `fs64` budget), one GPU. Every row is recorded with the
*quartile-mean* comparison described above rather than first-step-vs-last-step, and with
the parameter budget it actually trained — because a reward gain earned by 22 M free
parameters and one earned by 0.3 M adapters are different claims.

| model | mode | steps | trainable | reward early → late | verdict |
|---|---|---|---|---|---|
| `chronos-t5-small` | `grpo` (8 sampled continuations, group-relative advantage) | 96 | 1,081,344 (LoRA) | **1.741 → 1.839** | improved |
| `toto-2-22m` | `rft` (rejection-sampling filtered fine-tune) | 48 | 295,296 (LoRA) | −1.488 → −1.670 | **not improved** |
| `toto-2-22m` | `rft` | 48 | 21,915,584 (full) | −1.488 → −1.669 | **not improved** |
| `flowstate-r1` | `reward_descent` | 96 | heads_only at `bs=1 bf16` (ladder) | −1.296 → −1.387 | **not improved** |

### Repeated runs: what a reward delta is worth

`scripts/reward_seeds.py` runs the same protocol over several seeds and reports the
spread of the late-minus-early delta. Done properly (three seeds, and — for two of these
— on two different machines), the picture is more interesting than the single-curve
table above:

| model | mode | configuration | seeds | late − early | verdict |
|---|---|---|---|---|---|
| `chronos-t5-small` | `grpo` | LoRA 1.08 M, lr 2e-5, bs 8, 96 steps | 3 (A6000) | **+0.059 ± 0.013** | 3/3 improved |
| `flowstate-r1` | `reward_descent` | heads-only 1.18 M, **lr 2e-5** (the configuration of the table above), bs 1, 96 steps | 3 (GB10) + 3 (A6000) | +0.136 ± 0.348 | 1/3 — sign **not established** |
| `flowstate-r1` | `reward_descent` | heads-only, **lr 1e-4**, bs 1, 96 steps | 3 (GB10) + 3 (A6000) | **+0.426 ± 0.228** | 3/3 improved |
| `toto-2-22m` | `rft` | LoRA 295 k, default lr, bs 4, 48 steps | 3 (RTX Ada) | −0.067 ± 0.109 | 1/3 — no effect at this budget |

Three things follow, and they are the reason this section exists:

* **The FlowState row above ("not improved") was a single seed at lr 2e-5.** Repeated at
  that configuration it is +0.136 ± 0.348 with 1/3 seeds improved — i.e. the original
  −0.09 sits comfortably inside its own noise. At lr 1e-4 the same model and mode
  improves 3/3 with a mean five times the size of the earlier point estimate. So the
  honest claim is not "reward descent does not work on FlowState" but "**the reward
  trajectory at this budget is learning-rate sensitive**, and a single seed at one
  learning rate supports no verdict in either direction."
* **The Toto negative result survives**, with an error bar: −0.067 ± 0.109, 1/3 seeds.
  "RFT runs on Toto and does not improve reward at 48 steps" is supported; "RFT is
  useless for Toto" is not.
* **The harness is seed-deterministic across hardware.** The lr-2e-5 FlowState runs on
  the aarch64 GB10 and the x86 A6000 agree to five decimals (+0.42577 vs +0.42576 for
  the lr-1e-4 pair). Seed-to-seed variation is enormous; machine-to-machine variation is
  not. So repeated runs must vary the seed, and cross-machine replication is a check on
  the environment, not a source of new variance.

A reproduction caveat worth recording: the original demo runs were done on the DGX, where
the `toto` env now has a **CPU-only** torch (`2.7.0+cpu`) after its lock-file rebuild. A
GPU-flagged reproduction there fails with `Torch not compiled with CUDA enabled`, which is
environment drift, not a model result — the Toto seeds above were therefore run on the
laptop GPU, and the sweep now records each env's torch build and CUDA availability per row
(`docs/08_known_issues.md`).

The Toto row previously read **−2.232 → −1.914, improved**. It no longer does, and the
reason is worth writing down rather than quietly fixing. Two things moved since that
number was recorded: the reward's quantile-level grid (when the emitted head has a
different number of levels than the caller asked for, the candidate grid is now the
uniform one — the old scale is not comparable), and `peft="lora"` stopped training the
trunk (`docs/08_known_issues.md`). Re-run at the current revision with both a 295 k
adapter budget and the full 21.9 M-parameter model, the answer is the same in both
cases: the reward drifts down. So RFT is a working *path* on Toto — the mode runs, the
curve is logged, no fallback was needed — and a **negative result** at this budget.

The `chronos-t5-small` GRPO gain is the one positive claim here, and it survived the LoRA
fix, which strengthens it: +0.10 reward from 1.08 M adapter parameters, no ladder
fallback.

The FlowState row is kept as a negative result (re-run with quartile reporting: early
mean −1.296, late mean −1.387). What it shows: the run fell to the ladder profile
`bs=1, amp=bf16, peft=heads_only` under memory pressure, and gradient descent on the
pinball-plus-direction reward at that profile drifts the reward down instead of up. A reward *path* is demonstrated (the mode runs, the curve is logged);
a reward *gain* on a differentiable head is not demonstrated at this budget. Anyone
concluding "reward post-training helps TSForecasters" from this repo would be
over-reading it.

Reproduce:

```bash
tsfm-trl reward --model chronos-t5-small --steps 96 --bs 8 --lr 2e-5 \
                --budget fs64 --peft lora --device cuda
tsfm-trl reward --model toto-2-22m      --steps 48 --budget fs64 --peft lora --device cuda
```
