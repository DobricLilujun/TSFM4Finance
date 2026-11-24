# Compatibility matrix

Generated 2026-11-17 20:14:16 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moirai-1.1-R-small | A | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ✅ | 6.7 |
| moirai-2.0-R-small | A | ✅ | ✅ | ✅ | ✅ | · | ✅ (3.027→3.111) | ✅ | ✅ | ✅ | 2.3 |
| moirai-1.1-R-base | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ✅ | 6.2 |
| moirai-moe-1.0-R-base | B | ✅ | ✅ | ✅ | ✅ | · | ❌ | ✅ | ✅ | ❌ | 22.0 |

## Captured failures

- **moirai-1.1-R-small** (moirai, python 3.11.15): finetune: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
- **moirai-1.1-R-base** (moirai, python 3.11.15): finetune: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
- **moirai-moe-1.0-R-base** (moirai_moe, python 3.11.15): finetune: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn; reward: ValueError: peft='heads_only' matched no parameters; head_prefixes=('p21', 'quantile_head', 'head', 'output_proj', 'distribution_output'); declare them on the adapter (head_prefixes=...)
