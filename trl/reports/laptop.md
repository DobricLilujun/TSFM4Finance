# Compatibility matrix

Generated 2026-11-18 17:50:13 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see the JSON report for the captured reason) · — not offered by the model · ◐ partial · · not checked

Machines: ubadmin-Precision-3591/x86_64. One row per (model, environment); when two machines disagree the row that ran is kept and the other is recorded under *other machine* in the reasons.

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| timesfm-2.0-500m | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 36.7 |
| arima | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 8.4 |
| moving_average | A | ✅ | ✅ | ✅ | — | · | · | ✅ 2/2 | ✅ | · | 1.2 |
| naive | A | ✅ | ✅ | ✅ | — | · | · | ✅ 2/2 | ✅ | · | 1.4 |
| seasonal_naive | A | ✅ | ✅ | ✅ | — | · | · | ✅ 2/2 | ✅ | · | 1.5 |
| theta | A | ✅ | ✅ | ✅ | — | · | · | ✅ 2/2 | ✅ | · | 2.2 |
| chronos-2 | A | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ 2/2 | ✅ | ✅ | 9.2 |
| chronos-bolt-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (3.530→3.526) | ✅ 2/2 | ✅ | ✅ | 3.0 |
| chronos-t5-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (1.724→1.809) | ✅ 2/2 | ✅ | ✅ | 20.2 |
| flowstate-r1 | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (9.862→5.641) | ✅ 2/2 | ✅ | ✅ | 18.6 |
| timesfm-2.5-200m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (0.347→0.450) | ✅ 2/2 | ✅ | ✅ | 10.2 |
| ttm-r2 | A | ✅ | ✅ | ✅ | — | · | ✅ (260.703→310.215) | ✅ 2/2 | ✅ | ✅ | 2.1 |
| ttm-r3 | A | ✅ | ✅ | ✅ | — | · | ✅ (134.923→118.313) | ✅ 2/2 | ✅ | ✅ | 2.5 |
| chronos-bolt-base | B | ✅ | ✅ | ✅ | ✅ | · | ✅ (6.194→4.667) | ✅ 2/2 | ✅ | ✅ | 12.5 |
| chronos-t5-base | B | ✅ | ✅ | ❌ | · | · | ✅ (1.673→1.687) | ✅ 2/2 | ✅ | ❌ | 487.5 |
| timesfm-3.0 | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 29.4 |
| patchtst-fm-r2 | C | ✅ | ✅ | ✅ | — | · | ❌ | ✅ 2/2 | ✅ | ❌ | 9.0 |
| moirai-1.1-R-small | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 9.2 |
| moirai-2.0-R-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (2.650→3.201) | ✅ 2/2 | ✅ | ✅ | 3.7 |
| moirai-1.1-R-base | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 5.2 |
| moirai-moe-1.0-R-base | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 17.3 |
| moment-1-large | A | ✅ | ✅ | ✅ | — | · | ✅ (83.293→4.007) | ✅ 3/3 | ✅ | ✅ | 68.4 |
| moment-1-small | B | ✅ | ✅ | ✅ | — | · | ✅ (223.330→96.402) | ✅ 3/3 | ✅ | ✅ | 8.0 |
| tirex | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 41.1 |
| tirex-2 | B | ✅ | ❌ | · | · | · | · | · | · | · | 1.5958282947540283 |
| timemoe-50m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (0.025→0.043) | ✅ 1/2 | ✅ | ✅ | 110.4 |
| kronos-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (2.356→2.423) | ✅ 2/2 | ✅ | ✅ | 186.6 |
| kronos-base | B | ✅ | ✅ | ✅ | ✅ | · | ✅ (2.575→2.595) | ✅ 2/2 | ✅ | ✅ | 621.3 |
| kronos-mini | B | ✅ | ✅ | ✅ | ✅ | · | ✅ (9.017→9.076) | ✅ 2/2 | ✅ | ✅ | 51.9 |
| visiontspp-base | C | ❌ | ❌ | · | · | · | · | · | · | · | 1.3327279090881348 |
| hirondel-7m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (0.061→0.104) | ✅ 2/2 | ✅ | ✅ | 13.5 |
| kairos-10m | A | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ 2/2 | ✅ | ❌ | 6.2 |
| t0-gamma | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 7.1 |
| timestral-28m | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 14.8 |
| aurora | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 37.1 |
| kairos-50m | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ 2/2 | ✅ | ❌ | 5.7 |
| t0-alpha | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 5.5 |
| t0-beta | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 6.6 |
| timestral-123m | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 19.8 |
| timestral-5m | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 11.0 |
| yug | B | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 9.3 |
| timestral-209m | C | ✅ | ✅ | ✅ | ✅ | · | · | ✅ 2/2 | ✅ | · | 23.7 |
| toto-2-22m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (0.710→0.714) | ✅ 2/2 | ✅ | ✅ | 9.5 |
| toto-open-1.0 | B | ✅ | ✅ | ✅ | ✅ | · | ✅ (2.988→2.586) | ✅ 2/2 | ✅ | ✅ | 23.5 |
| toto-2-313m | B | ✅ | ✅ | ✅ | ✅ | · | ✅ (1.211→1.157) | ✅ 2/2 | ✅ | ✅ | 31.2 |

## Captured failures

- **timesfm-2.0-500m** (timesfm20, python 3.11.15): finetune: timesfm-2.0-500m: no 'finetune' support (adapter exposes no trainable module); reward: timesfm-2.0-500m: no 'finetune' support (adapter exposes no trainable module)
- **arima** (baseline, python 3.11.15): reward: arima: no 'finetune' support (adapter exposes no trainable module)
- **moving_average** (baseline, python 3.11.15): reward: moving_average: no 'finetune' support (adapter exposes no trainable module)
- **naive** (baseline, python 3.11.15): reward: naive: no 'finetune' support (adapter exposes no trainable module)
- **seasonal_naive** (baseline, python 3.11.15): reward: seasonal_naive: no 'finetune' support (adapter exposes no trainable module)
- **theta** (baseline, python 3.11.15): reward: theta: no 'finetune' support (adapter exposes no trainable module)
- **chronos-t5-base** (chronos_t5, python 3.11.15): predict: OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB. GPU 0 has a total capacity of 5.64 GiB of which 46.44 MiB is free. Process 915373 has 163.04 MiB memory in use. Including non-PyTorch memory, this process has 5.40 GiB memory in use. Of the allocated memory 4.92 GiB is allocated by P; reward: reward training failed even on the last profile (CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 5.64 GiB of which 22.44 MiB is free. Process 915373 has 163.04 MiB memory in use. Including non-PyTorch memory, this process has 5.42 GiB memory in use. Of the allocated memory 4.86 GiB is allocated by PyTorch, and 422.99 MiB is reserved by PyTorch but unallocated. If res
- **timesfm-3.0** (timesfm30, python 3.11.15): finetune: timesfm-3.0: no 'finetune' support (the public timesfm package ships TimesFM 3.0 inference only (no documented training forward); contributing a head-only loop is tracked in docs/03_fine_tune.md); reward: timesfm-3.0: no 'finetune' support (adapter exposes no trainable module); device: cpu (retried after a GPU out-of-memory on auto: UnavailableError: timesfm-3.0: OutOfMemoryError: CUDA out of memory. Tried to allocate 20.)
- **patchtst-fm-r2** (patchtst, python 3.11.15): finetune: TypeError: PatchTSTFMForPrediction.forward() got an unexpected keyword argument 'future_values'; reward: ValueError: peft='heads_only' matched no parameters; head_prefixes=('head', 'decoder'); declare them on the adapter (head_prefixes=...)
- **moirai-1.1-R-small** (moirai, python 3.11.15): finetune: moirai-1.1: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune); reward: moirai-1.1: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune)
- **moirai-1.1-R-base** (moirai, python 3.11.15): finetune: moirai-1.1: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune); reward: moirai-1.1: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune)
- **moirai-moe-1.0-R-base** (moirai_moe, python 3.11.15): finetune: moirai-moe: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune); reward: moirai-moe: no 'finetune' support (this uni2ts release exposes Moirai 1.1/MoE inference only through sample paths (``forward`` draws from the predicted distribution), and its ``_get_distr`` returns a distribution over the whole padded window, not the prediction slice. A differentiable step therefore needs uni2ts' own training stack (MoiraiForecast + PackedMaskedGlueTabDataset, see docs/03_fine_tune.md) — optimising quantiles of samples would be a no-op dressed as a fine-tune)
- **tirex** (tirex, python 3.11.15): reward: tirex: no 'finetune' support (adapter exposes no trainable module)
- **tirex-2** (tirex2, python 3.12.13): load: UnavailableError: tirex-2: UnavailableError: tirex-2: needs either the Triton kernels (GPU sm8.9 below sm10.0) or a CUDA toolchain for the native fallback (nvcc not found)  [install: run it on a Blackwell-class GPU (sm >= 10.0) with CUDA 13 + Triton >= 3.4, or install the full CUDA toolkit (nvcc) fo
- **timemoe-50m** (timemoe, python 3.11.15): finetune: loss did not fall: 0.024578 -> 0.042918; classify:probe: OutOfMemoryError: CUDA out of memory. Tried to allocate 92.00 MiB. GPU 0 has a total capacity of 5.64 GiB of which 8.44 MiB is free. Process 915373 has 163.04 MiB memory in use. Including non-PyTorch 
- **visiontspp-base** (visionts, python 3.11.15): load: UnavailableError: visiontspp-base: UnavailableError: visiontspp-base: VisionTS++ needs the upstream package (the Hub repo ships .ckpt weights only): No module named 'visiontspp'  [install: git clone the VisionTS/VisionTS++ upstream, then weights=/path/to/visiontspp_base.ckpt. clone it and point the 
- **hirondel-7m** (hirondel, python 3.12.13): finetune: loss did not fall: 0.061029 -> 0.103562
- **kairos-10m** (kairos, python 3.12.13): finetune: RuntimeError: Boolean value of Tensor with more than one value is ambiguous; reward: RuntimeError: grad can be implicitly created only for scalar outputs
- **t0-gamma** (t0, python 3.12.13): finetune: t0-gamma: no 'finetune' support (adapter exposes no trainable module); reward: t0-gamma: no 'finetune' support (adapter exposes no trainable module)
- **timestral-28m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **aurora** (aurora, python 3.12.13): finetune: aurora: no 'finetune' support (adapter exposes no trainable module); reward: aurora: no 'finetune' support (adapter exposes no trainable module)
- **kairos-50m** (kairos, python 3.12.13): finetune: RuntimeError: Boolean value of Tensor with more than one value is ambiguous; reward: RuntimeError: grad can be implicitly created only for scalar outputs
- **t0-alpha** (t0, python 3.12.13): finetune: t0-gamma: no 'finetune' support (adapter exposes no trainable module); reward: t0-gamma: no 'finetune' support (adapter exposes no trainable module)
- **t0-beta** (t0, python 3.12.13): finetune: t0-gamma: no 'finetune' support (adapter exposes no trainable module); reward: t0-gamma: no 'finetune' support (adapter exposes no trainable module)
- **timestral-123m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **timestral-5m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
- **yug** (yug, python 3.12.13): finetune: yug: no 'finetune' support (adapter exposes no trainable module); reward: yug: no 'finetune' support (adapter exposes no trainable module)
- **timestral-209m** (timestral, python 3.12.13): finetune: timestral: no 'finetune' support (the released timestral package exposes the inference forecaster only; wire the vendor training forward (mistralai/timestral) before claiming a Timestral fine-tune); reward: timestral: no 'finetune' support (adapter exposes no trainable module)
