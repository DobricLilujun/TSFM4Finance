# Compatibility matrix

Generated 2026-11-17 20:07:43 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moment-1-large | A | ✅ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 27.8 |
| moment-1-small | B | ❌ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 7.8 |

## Captured failures

- **moment-1-large** (moment, python 3.11.15): predict: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x16384 and 65536x16); finetune: RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x16384 and 65536x16); classify: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x15360 and 65536x64); anomaly: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x16384 and 65536x32); reward: RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x16384 and 65536x16)
- **moment-1-small** (moment, python 3.11.15): predict: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x8192 and 32768x16); finetune: RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x8192 and 32768x16); classify: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x7680 and 32768x64); anomaly: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x8192 and 32768x32); reward: RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x8192 and 32768x16)
