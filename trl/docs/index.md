# tsfm-trl

Time-series foundation models behind one API — and an honest map of what each
architecture can actually do in a given environment.

`tsfm-trl` is first a **benchmark and compatibility harness**: 46 open TSFM
checkpoints (Chronos family, TimesFM 2.x/2.5/3.0, Moirai 1.1/2.0/MoE, MOMENT, TTM,
Toto, Time-MoE, Sundial, Kronos, TiRex, Kairos, Timestral, Hirondel, t0, FlowState,
VisionTS, Aurora, Yug, plus statistical baselines) loaded through one adapter
contract, then driven through the same set of tasks. Post-training (fine-tune,
probing, reward/RL) sits on top of that contract — it is a capability gate, not a
claim that any model got better.

## The two claims this library keeps apart

| | question | produced by | can be quoted as |
|---|---|---|---|
| **compatibility** | does this architecture load, forecast, emit quantiles, embed, classify, detect anomalies, take post-training — *in this environment*? | `tsfm-trl doctor` → [measured matrix](matrix.md) | yes, with the machine and env in the row |
| **effectiveness** | how accurate is it? | `tsfm-trl evaluate` → RMSE/MASE/CRPS/WQL/coverage | yes, per the [protocol](09_benchmark.md) |
| **capability smoke** | does a training step run at all? | `doctor`'s fine-tune/reward columns | only as "it runs", never as "it helps" |

Mixing the third into the first is the most common way a TSFM table becomes
misleading, so the matrix labels it separately and the fine-tune cell says what its
numbers mean.

## Quick tour

```bash
pip install -e .                       # core only; per-family venvs: scripts/make_envs.sh

tsfm-trl list --tier A                 # what the registry knows about
tsfm-trl forecast --model chronos-t5-small --horizon 21 --md
tsfm-trl evaluate --model toto-2-22m --horizons 21,63,126 --max-windows 32

tsfm-trl classify --model moirai-2.0-R-small --label dir5 --routes probe,head,native
tsfm-trl reward  --model chronos-t5-small --steps 48 --peft lora --mode auto

tsfm-trl doctor --all-envs --json reports/matrix.json      # the compatibility matrix
tsfm-trl doctor --merge reports/*.json --md COMPATIBILITY.md
```

## Measured compatibility (2026-11-21, 46 models, two machines)

| task | ok | declined by the model | failed | not checked |
|---|---:|---:|---:|---:|
| load | 45 | 0 | 1 | 0 |
| forecast / predict | 44 | 0 | 1 | 1 |
| quantiles | 35 | 9 | 0 | 2 |
| embed | 44 | 0 | 1 | 1 |
| classify | 44 | 0 | 1 | 1 |
| anomaly | 44 | 1 | 0 | 1 |
| **reward / RL** | **30** | 15 | 0 | 1 |
| fine-tune (capability) | 29 | 9 | 0 | 8 |

The two red cells are named, not averaged away: **VisionTS++** (the Hub repo ships
`.ckpt` weights only, the upstream package is not installable as a package) and
**TiRex-2** (the vendor's Triton mLSTM scan kernel compiles at import and then fails
at runtime on CUDA 13 / sm100, and the native fallback's JIT-built extension does not
import — see [known issues](08_known_issues.md)).

`declined` means the model or its public package has no path for that task here
(TimesFM's build is inference-only, Moirai 1.1's `forward` samples instead of returning
differentiable quantiles). `failed` means something was attempted and broke, with the
reason captured in the row. History says to doubt `failed` before believing
`declined`: three "the model cannot be fine-tuned" verdicts in this project turned out
to be our bugs.

## Which post-training path each architecture exposes

| path | models | requirement |
|---|---|---|
| `grpo` | 2 (Chronos-T5 small/base) | token log-probabilities over its own sampled paths |
| `rft` | 18 | can *sample* candidate paths; no log-prob needed |
| `descent` | 5 | differentiable quantile head |
| `descent·pt` | 5 | differentiable point head only |
| — | 15 | no trainable path in the public build |

GRPO here is hand-written (~435 lines in `tsfm_trl/training/reward.py`), with no RL
library dependency; only two architectures expose the token log-probs that true GRPO
needs, which is a property of the vendor builds and not something the harness papers
over. Details in [RL / reward API](10_trl_api.md) and [Reward & RL paths](05_reward_rl.md).

## Where to read next

| if you want | go to |
|---|---|
| install and per-family environments | [Install](01_install.md) |
| a forecast, quantiles, coverage semantics | [Forecasting](02_forecast.md) |
| fine-tune / LoRA / budget rungs | [Fine-tune](03_fine_tune.md) |
| probing, heads, native classification | [Classification & probing](04_classify_probing.md) |
| reward, RL modes, what GRPO does and does not mean | [RL / reward API](10_trl_api.md), [Reward & RL](05_reward_rl.md) |
| how the matrix is produced and merged | [Compatibility matrix tool](06_matrix.md) |
| the measured table itself | [Measured matrix](matrix.md) |
| the benchmark protocol and metric list | [Benchmark protocol](09_benchmark.md) |
| finance-labelled tasks | [Finance tasks](07_finance_tasks.md) |
| what is broken and what we got wrong | [Known issues](08_known_issues.md) |

!!! note "About this site"
    The site is rendered from the same markdown that lives in the repository
    (`mkdocs.yml` + `scripts/build_docs.sh`); the [measured matrix](matrix.md) page is
    generated from `reports/compat_matrix.json` before each build, so the table cannot
    drift from the data.
