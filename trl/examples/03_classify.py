#!/usr/bin/env python3
"""Classification at three levels of invasiveness.
    python examples/03_classify.py [model] [dir|dir5|extreme|volreg]

Compare a TSFM's head against the frozen-feature probe and the statistical baseline —
otherwise a 58 % accuracy on a 60 %-majority task looks like skill.
"""
import sys

from tsfm_trl import ClassificationTrainer, ModelHub, TrainingConfig
from tsfm_trl.data.classify import direction_dataset, extreme_dataset, volreg_dataset
from tsfm_trl.data.fixtures import tiny_finance

key = sys.argv[1] if len(sys.argv) > 1 else "chronos-bolt-small"
task = sys.argv[2] if len(sys.argv) > 2 else "dir5"

ds = tiny_finance(900)
builders = {"dir": lambda d: direction_dataset(d, horizon=5, ctx=120, levels=3),
            "dir5": lambda d: direction_dataset(d, horizon=5, ctx=120, levels=5),
            "extreme": lambda d: extreme_dataset(d, horizon=5, ctx=120, k=2.0),
            "volreg": lambda d: volreg_dataset(d, horizon=10, ctx=120)}
cds = builders[task](ds)
print(cds.describe())

m = ModelHub.load(key, device="auto", ctx_cap=256, batch=4)
cfg = TrainingConfig(ctx=120, horizon=5, budget="fs64", steps=40, bs=16, seed=0,
                     out_dir="reports/ft_examples", tag="clf_example")

print(f"\nbase rate (majority class) = {cds.describe()['base_rate']:.3f} — anything near "
      f"that is not skill")
for route in ("probe", "head", "native"):
    res = ClassificationTrainer(m, cfg, route=route).fit(cds)
    met = res.extra.get("metrics") or {}
    tr_m = res.extra.get("train_metrics") or {}
    print(f"{route:7s} {res.status:7s} "
          f"test acc={met.get('accuracy', float('nan')):.3f} "
          f"macro_f1={met.get('macro_f1', float('nan')):.3f} "
          f"f1_β2={met.get('f1_beta2', float('nan')):.3f} "
          f"pr_auc={met.get('pr_auc', float('nan')):.3f}"
          + (f" | train acc={tr_m.get('accuracy', float('nan')):.3f}" if tr_m else "")
          + (f" | {res.reason}" if res.reason else ""))
print("\ntrain-vs-test gap is the point: a head that fits the train split perfectly and "
      "sits at base rate on test has memorised, not learned.")
