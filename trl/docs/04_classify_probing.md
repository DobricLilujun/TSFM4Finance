# 4 · Classification and probing

Three levels, from cheapest to most invasive. The plan requires all three to be
*checked* for every model, and the matrix shows which one a family actually supports.

```python
from tsfm_trl import ClassificationTrainer, ModelHub, TrainingConfig
from tsfm_trl.data.classify import direction_dataset, extreme_dataset, volreg_dataset, ucr_dataset

ds  = …                                   # a SeriesDataset (e.g. a price panel)
cds = direction_dataset(ds, horizon=5, ctx=120)
m   = ModelHub.load("moment-1-large", device="cuda")

ClassificationTrainer(m, TrainingConfig(budget="fs16"), route="probe").fit(cds)
ClassificationTrainer(m, TrainingConfig(budget="fs16", peft="lora"), route="head").fit(cds)
ClassificationTrainer(m, TrainingConfig(budget="fs16"), route="native").fit(cds)
```

| route | what trains | notes |
|---|---|---|
| `probe` | nothing inside the model | frozen representations + a classical classifier |
| `head` | a new readout head, optionally `+ lora` on the trunk | the default; `unfreeze_trunk=True` widens it |
| `native` | the family's own classification task/head | only where the vendor provides one (MOMENT's `classification` task, Moirai's `FORECASTING`-style heads) |

CLI:

```bash
tsfm-trl classify --model moment-1-large --task dir5 --levels probe,head,native
```

## Probe features

`tsfm_trl.probing.features` has two feature families, and the matrix reports them
separately because a statistical probe is a *baseline*, not a property of the TSFM:

* `stat_matrix(X)` — shape/scale-free summary statistics per window (level, trend
  slope, volatility, autocorrelation, crossing rate, tail statistics…). Any model,
  including none at all.
* `model_matrix(m, X)` — the model's own embeddings/pooled hidden states, produced by
  `m.embeddings(...)` / `m.classify_features(...)`. Models that expose no trunk
  representation decline with a `CapabilityError`, and the row shows `—`.

Both are evaluated with **expanding-window CV** (`folds(cds.split, mode="expanding")`):
financial windows leak forward otherwise. The classifier is logistic or gradient
boosting (`make_classifier`), and `classify_eval(model, cds, routes=("probe_stat",
"probe_model"), kinds=("logistic","gbm"))` returns one row per combination.

## Label tasks

`data.classify` builds label sets from a `SeriesDataset` so a classification row means
the same thing for every model:

| builder | task | default |
|---|---|---|
| `direction_dataset(ds, horizon, ctx, band=0.25)` | sign of the forward move, with a dead-band (`dir5`-style 5-level variant via `band`) | 2–5 classes |
| `extreme_dataset(ds, horizon, ctx, k=3.0)` | is the move beyond `k` conditional sigmas (`*_extreme`) | 2 classes |
| `volreg_dataset(ds, horizon, ctx, n_bins=3)` | forward volatility regime | 3 classes |
| `ucr_dataset(name, ctx)` | a tiny public UEA/UCR subset, for external validity | per dataset |

`ClassificationDataset` stores `X` (context arrays), `y`, and a **time-ordered
`split`** (0 train / 1 test) — never a random split: shuffled windows across a price
series are an information leak that makes every number in the table meaningless.

## What is refused

* Panel-only families (Kronos) cannot take plain 1-D contexts for a head; the row says
  `— (needs OHLCV panels)` — except Kronos, whose adapter writes the flat
  `open=high=low=close, volume=amount=0` panel its own `finetune_csv` format uses, so
  its model-feature probe does run and says so in `Pred.meta`.
* Models with no trunk output (TimesFM inference builds, TiRex) decline
  `model_matrix` with a `CapabilityError`; the statistical probe still runs, and the
  table labels it `stat`, so nobody mistakes it for a representation result.
* `classification_metrics` reports balanced accuracy, macro-F1 and per-class counts —
  a 90 %-majority task that scores 90 % accuracy is visible as a 0.5 F1.
