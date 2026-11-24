# Compatibility matrix

Generated 2026-11-17 20:07:06 on ubadmin-Precision-3591 (x86_64), python 3.11.15.

✅ ok · ❌ failed (see `reports/compat_matrix.json` for the captured reason) · — not offered by the model · ◐ partial · · not checked

| model | tier | weights | load | forecast | quantiles | embed | fine-tune | classify | anomaly | reward | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moirai-1.1-R-small | A | ❌ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 6.2 |
| moirai-2.0-R-small | A | ✅ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 0.2 |
| moirai-1.1-R-base | B | ❌ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 5.5 |
| moirai-moe-1.0-R-base | B | ❌ | ✅ | ❌ | · | · | ❌ | ❌ | — | ❌ | 38.8 |

## Captured failures

- **moirai-1.1-R-small** (moirai, python 3.11.15): predict: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1; finetune: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1; classify: RuntimeError: The size of tensor a (14) must match the size of tensor b (34) at non-singleton dimension 1; anomaly: RuntimeError: The size of tensor a (14) must match the size of tensor b (34) at non-singleton dimension 1; reward: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1
- **moirai-2.0-R-small** (moirai2, python 3.11.15): predict: RuntimeError: The size of tensor a (9) must match the size of tensor b (17) at non-singleton dimension 1; finetune: RuntimeError: The size of tensor a (9) must match the size of tensor b (17) at non-singleton dimension 1; classify: RuntimeError: The size of tensor a (8) must match the size of tensor b (17) at non-singleton dimension 1; anomaly: RuntimeError: The size of tensor a (8) must match the size of tensor b (17) at non-singleton dimension 1; reward: RuntimeError: The size of tensor a (9) must match the size of tensor b (17) at non-singleton dimension 1
- **moirai-1.1-R-base** (moirai, python 3.11.15): predict: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1; finetune: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1; classify: RuntimeError: The size of tensor a (14) must match the size of tensor b (34) at non-singleton dimension 1; anomaly: RuntimeError: The size of tensor a (14) must match the size of tensor b (34) at non-singleton dimension 1; reward: RuntimeError: The size of tensor a (16) must match the size of tensor b (34) at non-singleton dimension 1
- **moirai-moe-1.0-R-base** (moirai_moe, python 3.11.15): predict: RuntimeError: The size of tensor a (9) must match the size of tensor b (17) at non-singleton dimension 1; finetune: RuntimeError: The size of tensor a (9) must match the size of tensor b (17) at non-singleton dimension 1; classify: RuntimeError: The size of tensor a (8) must match the size of tensor b (17) at non-singleton dimension 1; anomaly: RuntimeError: The size of tensor a (8) must match the size of tensor b (17) at non-singleton dimension 1; reward: ValueError: peft='heads_only' matched no parameters; head_prefixes=('p21', 'quantile_head', 'head', 'output_proj', 'distribution_output'); declare them on the adapter (head_prefixes=...)
