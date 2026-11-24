# Compatibility matrix

Generated 2026-11-17 20:24:11 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moment-1-large | A | ✅ | ✅ | ✅ | — | · | ✅ (52.618→4.222) | ✅ | ✅ | — | 86.9 |
| moment-1-small | B | ✅ | ✅ | ✅ | — | · | ✅ (100.108→43.131) | ✅ | ✅ | — | 4.9 |

## Captured failures

- **moment-1-large** (moment, python 3.11.15): reward: moment-1-large: no 'rft' support (model exposes no quantiles or sample paths, so there is no predictive distribution to draw candidates from (wire the vendor's sampler into the adapter to enable RFT))
- **moment-1-small** (moment, python 3.11.15): reward: moment-1-large: no 'rft' support (model exposes no quantiles or sample paths, so there is no predictive distribution to draw candidates from (wire the vendor's sampler into the adapter to enable RFT))
