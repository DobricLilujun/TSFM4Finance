#!/usr/bin/env python3
"""Fine-tune, save, and re-load with provenance.
    python examples/02_finetune.py [model] [full|heads_only|lora]

The checkpoint is not just weights: it records the base model id, which parameters were
trainable, the config, the split boundary and the loss curve — enough to refuse a
wrongly paired checkpoint and to prove the run never saw the evaluation span.
"""
import sys

from tsfm_trl import ForecastTrainer, ModelHub, TrainingConfig, load_finetuned
from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.training.checkpoints import describe

key = sys.argv[1] if len(sys.argv) > 1 else "chronos-bolt-small"
peft = sys.argv[2] if len(sys.argv) > 2 else "lora"
ds = tiny_finance(600)

m = ModelHub.load(key, device="auto", ctx_cap=512, batch=4)
cfg = TrainingConfig(ctx=256, horizon=16, budget="fs16", steps=8, bs=4, peft=peft,
                     seed=0, out_dir="reports/ft_examples", tag="example")
res = ForecastTrainer(m, cfg).fit(ds)

print(f"status={res.status} steps={res.steps} trainable={res.trainable_parameters} "
      f"loss {res.first_loss:.4f} -> {res.final_loss:.4f} in {res.seconds:.1f}s")
print("fallback profiles used:", res.extra.get("fallbacks", []))
print("trainable (first 6):", res.trained_parameters[:6])

if res.checkpoint:
    print("\ncheckpoint provenance:")
    for k, v in describe(res.checkpoint).items():
        print(f"  {k}: {v}")
    fresh = ModelHub.load(key, device="auto", ctx_cap=512, batch=4)
    load_finetuned(fresh, res.checkpoint)
    print("reloaded into a fresh instance: ok")
    # and the refusal that makes provenance worth having:
    try:
        load_finetuned(ModelHub.load("moirai-2.0-R-small", device="auto"), res.checkpoint)
    except Exception as e:
        print("loading it onto a different family:", type(e).__name__, str(e)[:140])
