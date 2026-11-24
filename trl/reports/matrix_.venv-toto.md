# Compatibility matrix

Generated 2026-11-17 20:08:26 on ubadmin-Precision-3591 (x86_64), python 3.12.13.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toto-2-22m | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (1.198→1.190) | ✅ | ✅ | ✅ | 9.5 |
| toto-open-1.0 | B | ❌ | ❌ | · | · | · | · | · | · | · | 0.009804487228393555 |
| toto-2-313m | B | ❌ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ✅ | 32.6 |

## Captured failures

- **toto-open-1.0** (toto1, python 3.12.13): load: UnavailableError: toto-open-1.0: missing package: No module named 'transformers'
- **toto-2-313m** (toto2, python 3.12.13): finetune: RuntimeError: non-finite loss at step 1: nan
