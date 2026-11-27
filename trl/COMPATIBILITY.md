# Compatibility matrix

Generated 2026-11-21 16:33:48 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see the JSON report for the captured reason) · — not offered by the model · ◐ partial · · not checked

**Benchmark columns** (the point of this table): load · forecast · quantiles · embed · classify · anomaly. `classify n/m` counts the probe / head+LoRA / native-head routes that ran.

**reward (mode)** names the RL path the *architecture* allows — `grpo` (group-relative policy gradient over the model's own sampled token paths, needs log-probs), `rft` (rejection-sampling best-of-K, the group-baseline limit and the only path for sample-only heads), `descent` (gradient ascent on a differentiable quantile head), `descent·pt` (same through the point head). The trainer picks, not the caller; all four share one reward spec. It is a hand-written trainer with no RL library (`docs/05_reward_rl.md`).

**fine-tune (capability)** is a gate, not a result: it says a differentiable vendor objective ran, gradients were feasible and loss logging worked, over the **two** smoke steps shown as `(a→b)`. It is *not* evidence that fine-tuning improves out-of-sample error — that is a benchmark run with a metric, and the effectiveness question lives in `docs/09_benchmark.md`.

Machines: gpu7/aarch64, ubadmin-Precision-3591/x86_64. One row per (model, environment); when two machines disagree the row that ran is kept and the other is recorded under *other machine* in the reasons.

| model | tier | weights | load | forecast | quantiles | embed | classify | anomaly | reward (mode) | fine-tune (capability) | s | machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arima | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 5.9 | gpu7/aarch64 |
| moving_average | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | — | — | 0.7 | gpu7/aarch64 |
| naive | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | — | — | 0.6 | gpu7/aarch64 |
| seasonal_naive | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | — | — | 0.6 | gpu7/aarch64 |
| theta | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | — | — | 0.8 | gpu7/aarch64 |
| chronos-2 | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ | 14.3 | gpu7/aarch64 |
| chronos-bolt-small | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ descent | ✅ (3.530→3.526) | 2.4 | gpu7/aarch64 |
| chronos-t5-small | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ grpo | ✅ (1.774→1.814) | 10.3 | gpu7/aarch64 |
| flowstate-r1 | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ descent | ✅ (7.098→7.722) | 2.0 | gpu7/aarch64 |
| hirondel-7m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.061→0.104) | 13.5 | ubadmin-Precision-3591/x86_64 |
| kairos-10m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.458→0.110) | 8.8 | ubadmin-Precision-3591/x86_64 |
| kronos-small | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (2.344→2.453) | 63.5 | gpu7/aarch64 |
| moirai-1.1-R-small | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (2.562→2.413) | 14.0 | gpu7/aarch64 |
| moirai-2.0-R-small | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ descent | ✅ (2.650→3.201) | 1.6 | gpu7/aarch64 |
| moment-1-large | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 3/3 | ✅ | ✅ descent·pt | ✅ (219.576→97.814) | 30.4 | gpu7/aarch64 |
| sundial-base-128m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (3.436→3.505) | 20.4 | gpu7/aarch64 |
| t0-gamma | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (3.052→3.046) | 8.8 | ubadmin-Precision-3591/x86_64 |
| timemoe-50m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.025→0.043) | 148.6 | gpu7/aarch64 |
| timesfm-2.0-500m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 85.0 | gpu7/aarch64 |
| timesfm-2.5-200m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ descent | ✅ (0.347→0.450) | 5.9 | gpu7/aarch64 |
| timestral-28m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 14.8 | ubadmin-Precision-3591/x86_64 |
| tirex | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 17.1 | gpu7/aarch64 |
| toto-2-22m | A | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.710→0.714) | 26.1 | gpu7/aarch64 |
| ttm-r2 | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | ✅ descent·pt | ✅ (219.444→315.138) | 0.8 | gpu7/aarch64 |
| ttm-r3 | A | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | ✅ descent·pt | ✅ (134.459→118.437) | 1.4 | gpu7/aarch64 |
| aurora | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 37.1 | ubadmin-Precision-3591/x86_64 |
| chronos-bolt-base | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ descent | ✅ (3.606→3.812) | 7.3 | gpu7/aarch64 |
| chronos-t5-base | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ grpo | ✅ (1.702→1.733) | 29.3 | gpu7/aarch64 |
| kairos-50m | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.099→0.114) | 4.7 | ubadmin-Precision-3591/x86_64 |
| kronos-base | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (2.559→2.605) | 203.8 | gpu7/aarch64 |
| kronos-mini | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (9.066→9.096) | 26.1 | gpu7/aarch64 |
| moirai-1.1-R-base | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (2.576→2.852) | 6.0 | gpu7/aarch64 |
| moirai-moe-1.0-R-base | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (3.496→3.328) | 52.9 | gpu7/aarch64 |
| moment-1-small | B | ✅ | ✅ | ✅ | — | ✅ | ✅ 3/3 | ✅ | ✅ descent·pt | ✅ (244.804→108.576) | 5.3 | gpu7/aarch64 |
| t0-alpha | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (4.039→3.726) | 4.7 | ubadmin-Precision-3591/x86_64 |
| t0-beta | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (2.990→3.380) | 6.0 | ubadmin-Precision-3591/x86_64 |
| timesfm-3.0 | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 12.7 | gpu7/aarch64 |
| timestral-123m | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 19.8 | ubadmin-Precision-3591/x86_64 |
| timestral-5m | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 11.0 | ubadmin-Precision-3591/x86_64 |
| tirex-2 | B | ✅ | ✅ | ❌ | — | ❌ | ❌ 0/2 | — | — | — | 8.6 | gpu7/aarch64 |
| toto-open-1.0 | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (3.067→2.643) | 235.2 | gpu7/aarch64 |
| toto-2-313m | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | ✅ rft | ✅ (0.703→0.712) | 93.0 | gpu7/aarch64 |
| yug | B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 9.3 | ubadmin-Precision-3591/x86_64 |
| patchtst-fm-r2 | C | ✅ | ✅ | ✅ | — | ✅ | ✅ 2/2 | ✅ | ✅ descent·pt | — | 21.6 | gpu7/aarch64 |
| timestral-209m | C | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 2/2 | ✅ | — | — | 23.7 | ubadmin-Precision-3591/x86_64 |
| visiontspp-base | C | ❌ | ❌ | — | — | — | — | — | — | — | 1.2 | gpu7/aarch64 |

## Summary

| task | ok | declined by the model | failed | not checked |
|---|---|---|---|---|
| load | 45 | 0 | 1 | 0 |
| predict | 44 | 1 | 1 | 0 |
| quantiles | 35 | 11 | 0 | 0 |
| embed | 44 | 1 | 1 | 0 |
| classify | 44 | 1 | 1 | 0 |
| anomaly | 44 | 2 | 0 | 0 |
| reward | 30 | 16 | 0 | 0 |
| finetune | 29 | 17 | 0 | 0 |

'declined' means the model or its vendor package offers no path for that task in this environment (a `—` in the table): TimesFM's and TiRex's public builds ship inference only, Timestral and Yug likewise, Aurora's flow-matching head has no public training step, and the installed tsfm_public `PatchTSTFMForPrediction.forward` declares no target argument. 'failed' means something was attempted and broke, with the reason captured below. A declined row is a statement about the model package; a failed row is a statement about this environment or our adapter — and history says to check the second category before believing the first (`docs/08_known_issues.md`, fifth pass).


## RL reachability

Which post-training path each architecture exposes. `grpo` is the strictest (token log-probs on its own sampled paths); `rft` is what a sample-only or diffusion head can do; `descent` needs a differentiable head; the point-only models get the point version. This is a property of the vendor's public build, not a preference of the harness.

| path | models |
|---|---|
| grpo | 2: chronos-t5-base, chronos-t5-small |
| rft | 18: chronos-2, hirondel-7m, kairos-10m, kairos-50m, kronos-base, kronos-mini, kronos-small, moirai-1.1-R-base, moirai-1.1-R-small, moirai-moe-1.0-R-base, sundial-base-128m, t0-alpha, t0-beta, t0-gamma, timemoe-50m, toto-2-22m, toto-2-313m, toto-open-1.0 |
| descent | 5: chronos-bolt-base, chronos-bolt-small, flowstate-r1, moirai-2.0-R-small, timesfm-2.5-200m |
| descent·pt | 5: moment-1-large, moment-1-small, patchtst-fm-r2, ttm-r2, ttm-r3 |

declined by the package (no trainable path at all): 16 · not checked: 0


## Captured failures

- **arima** (baseline, python 3.11.14): reward: arima: no 'finetune' support (adapter exposes no trainable module); finetune: baseline: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **moving_average** (baseline, python 3.11.14): reward: moving_average: no 'finetune' support (adapter exposes no trainable module); finetune: baseline: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **naive** (baseline, python 3.11.14): reward: naive: no 'finetune' support (adapter exposes no trainable module); finetune: baseline: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **seasonal_naive** (baseline, python 3.11.14): reward: seasonal_naive: no 'finetune' support (adapter exposes no trainable module); finetune: baseline: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **theta** (baseline, python 3.11.14): reward: theta: no 'finetune' support (adapter exposes no trainable module); finetune: baseline: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **chronos-2** (chronos2, python 3.11.14): other machine: gpu7/aarch64: reward=fail
- **hirondel-7m** (hirondel, python 3.12.13): finetune: loss did not fall: 0.061029 -> 0.103562
- **kairos-10m** (kairos, python 3.12.13): other machine: ubadmin-Precision-3591/x86_64: finetune=fail, reward=fail
- **moment-1-large** (moment, python 3.11.14): other machine: ubadmin-Precision-3591/x86_64: reward=fail
- **timemoe-50m** (timemoe, python 3.11.14): finetune: loss did not fall: 0.024578 -> 0.042918
- **timesfm-2.0-500m** (timesfm20, python 3.11.14): finetune: timesfm-2.0-500m: no 'finetune' support (adapter exposes no trainable module); reward: timesfm-2.0-500m: no 'finetune' support (adapter exposes no trainable module)
- **timestral-28m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **tirex** (tirex, python 3.11.14): reward: tirex: no 'finetune' support (adapter exposes no trainable module); finetune: tirex: no trainable module and no loss_kind in the installed package (nothing to fine-tune)
- **aurora** (aurora, python 3.12.13): finetune: aurora: no 'finetune' support (adapter exposes no trainable module); reward: aurora: no 'finetune' support (adapter exposes no trainable module)
- **chronos-t5-base** (chronos_t5, python 3.11.14): other machine: ubadmin-Precision-3591/x86_64: predict=fail, reward=fail
- **kairos-50m** (kairos, python 3.12.13): other machine: ubadmin-Precision-3591/x86_64: finetune=fail, reward=fail
- **timesfm-3.0** (timesfm30, python 3.11.14): finetune: timesfm-3.0: no 'finetune' support (the public timesfm package ships TimesFM 3.0 inference only (no documented training forward); contributing a head-only loop is tracked in docs/03_fine_tune.md); reward: timesfm-3.0: no 'finetune' support (adapter exposes no trainable module)
- **timestral-123m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **timestral-5m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **tirex-2** (tirex2, python 3.12.14): predict: CompilationError: at 85:21:
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH

        # load initial states
        matC_k_val = tl.load(matCinitial_ptr, boundary_check=(0, 1)).to(tl.float32)
        vecN; embed: CompilationError: at 85:21:
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH

        # load initial sta; other machine: ubadmin-Precision-3591/x86_64: load=fail; anomaly: CompilationError: at 85:21:
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH

        # load initial sta; reward: tirex-2: no 'finetune' support (adapter exposes no trainable module); finetune: tirex2: no trainable module and no loss_kind in the installed package (nothing to fine-tune); quantiles: not observed: the point forecast itself failed
- **toto-open-1.0** (toto1, python 3.12.14): other machine: gpu7/aarch64: load=fail
- **yug** (yug, python 3.12.13): finetune: yug: no 'finetune' support (adapter exposes no trainable module); reward: yug: no 'finetune' support (adapter exposes no trainable module)
- **patchtst-fm-r2** (patchtst, python 3.11.14): finetune: patchtst-fm-r2: no 'finetune' support (forward(['output_hidden_states', 'pad_mask', 'past_observed_mask', 'past_values', 'prediction_length', 'quantile_levels', 'return_dict', 'return_loss']) declares no target argument in transformers 5.22.1); other machine: gpu7/aarch64: finetune=fail
- **timestral-209m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **visiontspp-base** (visionts, python 3.11.14): load: UnavailableError: visiontspp-base: UnavailableError: visiontspp-base: VisionTS++ needs the upstream package (the Hub repo ships .ckpt weights only): No module named 'visiontspp'  [install: git clone the VisionTS/VisionTS++ upstream, then weights=/path/to/visiontspp_base.ckpt. clone it and point the ; other machine: ubadmin-Precision-3591/x86_64: load=fail; predict: not attempted: the model did not load; finetune: not attempted: the model did not load; classify: not attempted: the model did not load; anomaly: not attempted: the model did not load; reward: not attempted: the model did not load; quantiles: not attempted: the model did not load; embed: not attempted: the model did not load
