"""Minimal reward post-training: one token model, one quantile model.

    python examples/04_reward.py            # both, on GPU if present
    python examples/04_reward.py grpo       # just the GRPO path

Both runs are small (16 windows, ~1 minute) and log a reward curve, and both models are
in the ``core`` environment. ``toto-2-22m`` is the third demonstrator (RFT through its
inference path); it needs the ``toto`` environment — see ``docs/05_reward_rl.md`` for
all three curves, including the one that did not improve. The point is the
shape of the call, not the result: a reward run reports which *mode* it used
(`grpo` / `reward_descent` / `reward_descent_point` / `rft`) and whether the late steps
beat the early ones. Read the "not improved" verdicts in docs/05_reward_rl.md before
drawing conclusions from the numbers.
"""
import json

from tsfm_trl import ModelHub, RewardTrainer, TrainingConfig
from tsfm_trl.data.fixtures import tiny_finance
from tsfm_trl.training.reward import RewardSpec

device = "cuda"
only = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else None

cases = [
    # (model, why it is here)
    ("chronos-t5-small", "token model -> GRPO over sampled continuations"),
    ("flowstate-r1", "differentiable quantile head -> reward descent"),
]

ds = tiny_finance(400)
for model_name, why in cases:
    if only and only not in ("all", model_name.split("-")[0]):
        continue
    model = ModelHub.load(model_name, device=device, ctx_cap=512, batch=8)
    cfg = TrainingConfig(ctx=256, horizon=16, steps=48, bs=4, budget="fs64",
                         peft="lora", device=device, out_dir="/tmp/trl_reward",
                         tag=model_name, log_every=12)
    tr = RewardTrainer(model, cfg, spec=RewardSpec(w_pinball=1.0, w_dir=0.5, w_cross=0.2),
                       kl=0.01, n_candidates=8)
    res = tr.fit(ds)
    print(json.dumps({
        "model": model_name, "why": why,
        "status": res.status, "mode": res.extra.get("mode"), "steps": res.steps,
        "reward_early_mean": res.extra.get("reward_early_mean"),
        "reward_late_mean": res.extra.get("reward_late_mean"),
        "improved": res.extra.get("improved"),
        "fallbacks": res.extra.get("fallbacks"),
        "checkpoint": res.checkpoint,
    }, indent=1))
