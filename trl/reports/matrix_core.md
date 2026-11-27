# Compatibility matrix

Generated 2026-11-17 19:59:18 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arima | A | ✅ | ✅ | ✅ | ✅ | · | · | ✅ | ✅ | ❌ | 10.0 |
| moving_average | A | ✅ | ✅ | ✅ | — | · | · | ✅ | — | ❌ | 0.0 |
| naive | A | ✅ | ✅ | ✅ | — | · | · | ✅ | — | ❌ | 0.0 |
| seasonal_naive | A | ✅ | ✅ | ✅ | — | · | · | ✅ | — | ❌ | 0.0 |
| theta | A | ✅ | ✅ | ✅ | — | · | · | ✅ | — | ❌ | 1.1 |
| chronos-2 | A | ✅ | ✅ | ✅ | ✅ | · | ✅ | ✅ | ✅ | ✅ | 8.8 |
| chronos-bolt-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (5.451→5.430) | ✅ | ✅ | ✅ | 3.1 |
| chronos-t5-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (1.476→1.617) | ✅ | ✅ | ✅ | 37.2 |
| flowstate-r1 | A | ✅ | ✅ | ✅ | — | · | ❌ | ❌ | — | ❌ | 0.6 |
| timesfm-2.0-500m | A | ❌ | ❌ | · | · | · | · | · | · | · | 0.04083967208862305 |
| timesfm-2.5-200m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (0.587→0.612) | ✅ | ✅ | ✅ | 19.8 |
| ttm-r2 | A | ✅ | ✅ | ✅ | — | · | ✅ (243.700→347.199) | ✅ | — | ❌ | 1.1 |
| ttm-r3 | A | ❌ | ✅ | ✅ | — | · | ✅ (171.300→153.279) | ✅ | — | ❌ | 4.4 |
| chronos-bolt-base | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ✅ | 28.6 |
| chronos-t5-base | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ❌ | 2119.9 |
| timesfm-3.0 | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ❌ | 45.2 |
| patchtst-fm-r2 | C | ❌ | ❌ | · | · | · | · | · | · | · | 0.012737035751342773 |

## Captured failures

- **arima** (baseline, python 3.11.15): reward: arima: no 'finetune' support (adapter exposes no trainable module)
- **moving_average** (baseline, python 3.11.15): anomaly: CapabilityError: moving_average: no 'anomaly_head' support; reward: moving_average: no 'finetune' support (adapter exposes no trainable module)
- **naive** (baseline, python 3.11.15): anomaly: CapabilityError: naive: no 'anomaly_head' support; reward: naive: no 'finetune' support (adapter exposes no trainable module)
- **seasonal_naive** (baseline, python 3.11.15): anomaly: CapabilityError: seasonal_naive: no 'anomaly_head' support; reward: seasonal_naive: no 'finetune' support (adapter exposes no trainable module)
- **theta** (baseline, python 3.11.15): anomaly: CapabilityError: theta: no 'anomaly_head' support; reward: theta: no 'finetune' support (adapter exposes no trainable module)
- **flowstate-r1** (flowstate, python 3.11.15): finetune: RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_addmm); classify: RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_addmm); anomaly: RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_addmm); reward: flowstate-r1: no 'rft' support (model exposes no quantiles or sample paths, so there is no predictive distribution to draw candidates from (wire the vendor's sampler into the adapter to enable RFT))
- **timesfm-2.0-500m** (timesfm20, python 3.11.15): load: UnavailableError: timesfm-2.0-500m: AttributeError: module 'timesfm' has no attribute 'TimesFmHparams'
- **ttm-r2** (ttm, python 3.11.15): anomaly: CapabilityError: ttm-r2: no 'anomaly_head' support; reward: ttm-r2: no 'rft' support (model exposes no quantiles or sample paths, so there is no predictive distribution to draw candidates from (wire the vendor's sampler into the adapter to enable RFT))
- **ttm-r3** (ttm, python 3.11.15): anomaly: CapabilityError: ttm-r2: no 'anomaly_head' support; reward: ttm-r2: no 'rft' support (model exposes no quantiles or sample paths, so there is no predictive distribution to draw candidates from (wire the vendor's sampler into the adapter to enable RFT))
- **chronos-bolt-base** (chronos_bolt, python 3.11.15): finetune: RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_addmm)
- **chronos-t5-base** (chronos_t5, python 3.11.15): finetune: RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select); reward: reward training failed even on the last profile (CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 5.64 GiB of which 20.44 MiB is free. Process 915373 has 163.04 MiB memory in use. Including non-PyTorch memory, this process has 5.42 GiB memory in use. Of the allocated memory 4.89 GiB is allocated by PyTorch, and 402.61 MiB is reserved by PyTorch but unallocated. If res
- **timesfm-3.0** (timesfm30, python 3.11.15): finetune: timesfm-3.0: no 'finetune' support (the public timesfm package ships TimesFM 3.0 inference only (no documented training forward); contributing a head-only loop is tracked in docs/03_fine_tune.md); reward: timesfm-3.0: no 'finetune' support (the public timesfm package ships TimesFM 3.0 inference only (no documented training forward); contributing a head-only loop is tracked in docs/03_fine_tune.md)
- **patchtst-fm-r2** (patchtst, python 3.11.15): load: UnavailableError: patchtst-fm-r2: missing package: No module named 'tsfm_public.models.patchtst'
