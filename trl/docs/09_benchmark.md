# 09 · Running this as a benchmark

The compatibility matrix answers **"does this architecture work here at all"**. A
benchmark answers a different question: **"how well does it forecast, and how do the
architectures compare"**. Both run through the same API, and conflating them is the most
common way to over-claim from this repo, so the two are kept formally separate here.

## The two claims, in one table

| | question | command | what a ✅ / a number means |
|---|---|---|---|
| **compatibility** | can the architecture be loaded, forecast, be scored, and be reached by a training/RL path in *this* environment | `tsfm-trl doctor` → `COMPATIBILITY.md` | the code ran end-to-end at smoke size; nothing about accuracy |
| **effectiveness** | how accurate is it, and how does it compare across families, horizons, domains | `tsfm-trl evaluate` (`forecast_eval`) | numbers on held-out windows with a defined metric, split and seed |

`COMPATIBILITY.md` deliberately contains **no accuracy numbers**. Its fine-tune column
carries a two-step loss for one reason only: to prove the objective is differentiable and
the logging works. Reading it as "fine-tuning helped" is a category error
(`docs/03_fine_tune.md`).

## Metrics that exist (all hand-written, all NaN-safe)

`tsfm_trl/eval/metrics.py`, on the canonical `LEVELS` grid unless stated otherwise:

| family | function | note |
|---|---|---|
| point | `rmse`, `mae`, `mape`, `smape`, `nd` | scale-free ones (`nd`, `smape`) for cross-series comparison |
| scale-free skill | `mase`, `relative_skill` | both need the training span and a `season` (1 for the finance panels) |
| direction | `directional_accuracy` | anchor-relative: the last observed value by default, which is the version that matters for the finance labels (`docs/07_finance_tasks.md`) |
| ranking | `rank_corr` | per-window Spearman, useful for signal-style tasks |
| probabilistic | `pinball`, `crps_from_quantiles`, `wql`, `quantile_coverage` | quantile-based; **coverage only counts bands the model actually emits** — a model that ships 9 levels is not scored on 21 |
| classification | `classification_report` | PR-AUC (multiclass OVR), ROC-AUC, macro-F1, F2-beta, precision/recall, base rate; hard labels give `nan` for PR/ROC rather than a fabricated score |

`forecast_report(y, pred, train)` bundles the point + probabilistic set into one dict;
`forecast_eval(...)` produces one row per (span, horizon).

## Protocol

The library fixes the parts of a protocol that would otherwise differ per family:

* **one quantile grid** (`LEVELS`, 21 levels) — adapters that emit fewer levels are
  mapped onto their own subset and `Pred.meta["levels_emitted"]` records it;
* **one normalisation of the question** — anchor-relative targets for the finance labels,
  and the model's own scaler is never re-implemented outside the adapter;
* **context handling** — `ctx_step`/`max_context` per adapter, left-padding with the
  model's own pad mask instead of zeros (Moirai, FlowState) — a silent zero-pad changes
  the score for patch models;
* **split discipline** — `SeriesDataset.splits`, and any checkpoint carries
  `split_end` so `load_finetuned` can refuse an evaluation window that overlaps training.

Budget names in `TrainingConfig` are also evaluation sizes:

```python
BUDGETS = {"fs16": 16 windows / 40 steps / lr 1e-5,
           "fs64": 64 windows / 120 steps / lr 1e-5,
           "full": all windows / 3 epochs / lr 5e-6}
```

## Running an effectiveness pass

One model, several horizons, JSON rows out (stdout is a data channel — progress lines go
to stderr):

```bash
tsfm-trl evaluate --model chronos-2 --series px.csv --horizons 21,63,126 --ctx 512
tsfm-trl evaluate --model chronos-2 --adaptation reports/ft/chronos-2__…/model.pt \
                  --series px.csv --horizons 21,63
```

The second form is the zero-shot-vs-adapted comparison: the checkpoint's provenance
(base id, peft level, trained parameter names, `split_end`) is validated on load, so the
comparison cannot silently swap a TimesFM head onto a Moirai trunk or evaluate on
windows the run trained on.

The whole registry at once, one interpreter per family, one row per model per series per
horizon — this is what produced `EFFECTIVENESS.md` (rendered as the site's *Measured effectiveness* page):

```bash
# canonical series: fev-style directories with real train/validation/test splits and
# OHLCV covariates (panel-consuming models such as Kronos need them)
scripts/sweep_evaluate.py \
    --series data/open/sp500_long --series data/open/audusd --series data/open/btc_crypto \
    --horizons 21,63,126 --max-windows 12 --spans all \
    --out reports/effectiveness_dgx.jsonl --fresh

# merge several machines/runs; merging is per (model, series, span, horizon), so partial
coverage from different machines unions instead of overwriting, and every row keeps the
machine/interpreter/torch stamp of the run that produced it
scripts/sweep_evaluate.py --report reports/effectiveness_dgx.jsonl,\
reports/effectiveness_watson.jsonl,reports/effectiveness_laptop_vendor.jsonl \
    --md EFFECTIVENESS.md
```

The sweep refuses to run a model in an environment the registry does not name for it (a
missing env is a recorded skip, not a `core` fallback), writes rows to JSONL as they
land, and stamps each run with its machine, series, spans and window count — so a
published table can always be traced to the interpreter that produced each row.

Reward runs get the same treatment: `scripts/reward_seeds.py --model … --seeds 0,1,2`
reports the spread of the late-minus-early reward delta instead of one curve.

Through the API, which is what a sweep should use:

```python
from tsfm_trl import ModelHub
from tsfm_trl.eval.evaluate import forecast_eval

m = ModelHub.load("moirai-1.1-R-small", device="cuda", ctx_cap=512, batch=16)
rows = forecast_eval(m, "panels/px.csv", horizons=(21, 63, 126),
                     spans=("test",), ctx=512, max_windows=64, seed=0, season=1)
```

Across environments, use the same subprocess dispatch the matrix uses — one interpreter
per family (`envs/local.json`), because the family pins are mutually contradictory
(`docs/01_install.md`):

```bash
tsfm-trl doctor --all-envs --tasks load,predict,quantiles \
                --json reports/matrix_bench_core.json
```

`predict` is the benchmark's inference half: every row that is ✅ there can be dropped
into `forecast_eval` without further adaptation work.

## Reporting rules (what goes in a table, what does not)

1. **Report the metric family the model can serve.** A point-only head (TTM r2/r3) has no
   CRPS; write `—`, do not integrate a degenerate distribution. Same reason TiRex's
   NPP and Chronos-Bolt's samples both get empirical quantiles first: different routes to
   the same grid, then the same metric.
2. **Report coverage per emitted band** and say which bands (`Pred.meta`).
3. **One row per (model, environment)**; a model that is ✅ on one machine and OOM on
   another is a fact about the machine, and the matrix keeps both under *other machine*.
4. **State the parameter budget for adapted rows** — `peft="lora"` now means a frozen
   trunk, and `TrainResult.trainable_parameters` is the number to print
   (`docs/08_known_issues.md` explains why that sentence exists).
5. **Multiple seeds and windows.** A single 16-window pass is a smoke test; the harness
   defaults are deliberately smoke-sized (`fs16`/`fs64`) so an accidental 10-hour run is
   impossible by default.
6. **Baselines in the same table.** `naive`, `seasonal_naive`, `theta`, `arima`,
   `moving_average` are registered models with their own rows, so every effectiveness
   table can carry skill relative to them (`relative_skill`, MASE) rather than raw loss.

## What pass 1 shows (and what it does not license)

43 of 46 registry models have rows; the three that do not are named in the table with
their reason (TimeMoE ran out of a 6 GB laptop GPU, TiRex-2's mLSTM kernels fail on this
toolchain, VisionTS++ ships weights with no installable package). All numbers are
zero-shot, one seed, 12 windows per pass, on three daily series.

* **No published checkpoint beats the trivial scaler on point accuracy at these
  horizons.** On `sp500_long` every model — statistical baseline or foundation model —
  has MASE above ~4 at h=21 and above ~9 at h=126. MASE normalises by the in-sample mean
  |Δ| of the training span, so this says: at three-week and half-year horizons on a daily
  index, the level scale dominates and none of these weights produce point-forecast skill
  over "tomorrow ≈ today".
* **Directional accuracy sits at 0.5.** The best cells in the `dir-acc` column are
  ~0.53–0.59 and many are below 0.5, which is what noise looks like. For a finance
  benchmark that is the headline result, not a caveat.
* **What does separate the models is calibration.** On `sp500_long` at h=21 TiRex's
  nominal 80% band covers 0.81 (close to nominal), Chronos-T5-base's covers 0.74, and
  ARIMA's covers 0.98 — ARIMA's intervals are simply too wide. At h=126 on
  `btc_crypto`, coverage collapses for the same models (TiRex 0.27, ARIMA 0.30): long
  crypto bands from these weights are not usable as stated. A point-only model cannot
  even be charged with this failure, which is why the `pt` marker matters.
* **Ranking is not stable across series or horizon.** TiRex and Moirai-1.1-R-base lead on
  the index; Moirai-2.0-R-small and t0-gamma lead on crypto at h=126. Anyone quoting "X
  is the best TSFM" from one dataset is quoting noise.
* **The benchmark found a library bug the compatibility matrix could not.** TiRex's
  roll-forward path for horizons longer than one vendor output block mixed the batch axis
  into a 1-D context and raised — at `h ≤ 63` (the matrix grid) that code never ran, so
  the model "worked". It is now fixed and measurable, and it happens to be the best
  zero-shot point model on the index. A green `✅` in a capability matrix is evidence the
  call returned, not that the numbers are right.

What this does **not** license: any statement about a family being better in general
(12 windows, one seed, three daily series), anything about *adapted* checkpoints (not
run), or the word "significant" (no variance estimate exists for a forecast row).

## What is *not* done yet (so nobody reads it in)

* The effectiveness sweep that *has* been run is zero-shot only: 46 registry models ×
  3 canonical series (`sp500_long`, `audusd`, `btc_crypto`) × horizons 21/63/126, 12
  windows per pass, one seed, on three machines (DGX Spark aarch64, an x86 node with an
  RTX A6000, and a 6 GB laptop for the vendor-family models). The result lives in
  `EFFECTIVENESS.md` (rendered as the site's *Measured effectiveness* page).
* What that does **not** cover: more than one seed (so no confidence intervals on any
  cell), more datasets or frequencies (three daily series is a start, not GIFT-Eval),
  adapted checkpoints (`--spans test` + `--adaptation <ckpt>`, never `all`), and any
  statement that a family is "better" — with one seed and twelve windows a MASE ordering
  is a measurement, not a verdict.
* No multi-seed confidence intervals, no Diebolt–Mariano / MZ statistics, no horizon-grid
  plots.
* The finance label set (`dir5`, `extreme`) and a tiny public UEA/UCR subset are wired for
  classification; external-validity classification numbers across all 46 models are not
  in the repo either.
* Fine-tune effectiveness (does adaptation improve out-of-sample skill, per family, at a
  stated parameter budget) is a separate experiment that the checkpoint provenance format
  is designed to make auditable, and has not been run at benchmark scale.

If a paper table is the goal, the honest structure is: **compatibility matrix** (this
repo, complete), **effectiveness table** (to run: `evaluate` across families × horizons ×
budgets, with baselines), **adaptation table** (zero-shot vs LoRA at a stated budget, for
the 29 families where the training path exists).
