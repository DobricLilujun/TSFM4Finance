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
  the score for patch models; and `min_context` for checkpoints that declare a fixed
  input length (Granite/TTM): the evaluation window is raised to it, or padded with the
  oldest observation when the span cannot supply it, because feeding a shorter window
  makes those models pad themselves and scale the answer by the padded fraction;
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

### Sampling noise (measured 2026-11-27, four arena series, same bytes, same code)

`--max-windows` draws the test windows **uniformly at random**; same seed, same windows. At the
fleet default of 8 windows per cell, seed 0 vs seed 1 moves naive's own MASE by ~55% median
(max 100%), a model's MASE by ~54%, and the paired model/naive ratio by ~12% (max 25%). At 32
windows: ~21%, ~13%, ~7%. So aggregate shares over hundreds of cells are trustworthy, a single
cell within ~12% of parity is not a win or a loss, and `n_windows` + lane `seed` are part of the
merge protocol signature (`proto`) and a hard precondition in `cross_machine_check.py`.

**The noise does not average out symmetrically, because the headline statistic is a threshold.**
Pass 7 re-scored the same 66 series on the same node at 32 windows: *every* model's beat-naive share
fell (`chronos-bolt-small` 42.4% -> 15.2%, `arima` 56.1% -> 40.9%), the head-to-head
`chronos-bolt-small > chronos-2` moved 53.8% -> 40.2%, and **32.3% of cells change winner** between
the two window counts. Cause: the per-cell model/naive ratio is centred just *above* 1 with a wide
spread at 8 windows, so counting cells below 1.0 collects more false wins than true ones. Table and
mechanism in docs/08 ("The beat-naive share is a threshold on a noisy number"); reproduce with
`probes/window_sensitivity.py`. The 64-window pass says where it stops: 32w vs 64w shares differ by at
most 8 pp, so 32 is enough. Consequence for this document: **every share in the tables below is an
8-window number and is labelled as such**; any pass whose numbers will be quoted runs at
`--max-windows 32` (now the default of `sweep_evaluate.py`).

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

## What pass 3 shows (the whole `standard/` corpus: 101 series, 3,628 cells)

Scope: every one of the 101 independent per-instrument series in `standard/`, 19 models,
horizons 21 and 63, 8 windows per series, one seed, zero-shot. Series were split 66/35
between trux (2× RTX 6000 Ada, `core`+`moment`+`timemoe` envs) and watson (one A6000 via
SLURM, `core` only), disjoint, allocated 2:1 inside every dataset family so neither
machine's table is confounded with one family. Merged per `(model, series, span, horizon)`.

Median MASE relative to the naive scaler on the same cell (1.00 = equal), share of cells
where the model beat naive:

| model | cells | beat naive | med MASE/naive | med MASE | cov@40 | ctx |
|---|---:|---:|---:|---:|---:|---:|
| `arima` | 202 | **55%** | 1.00 | 4.50 | 0.47 | 256 |
| `chronos-2` | 202 | 40% | 1.02 | 4.66 | 0.38 | 256 |
| `chronos-bolt-small` | 202 | 41% | 1.03 | 4.73 | 0.38 | 256 |
| `theta` | 202 | 30% | 1.04 | 4.79 | — | 256 |
| `chronos-bolt-base` | 202 | 32% | 1.04 | 4.99 | 0.39 | 256 |
| `chronos-t5-base` | 202 | 35% | 1.05 | 4.99 | 0.33 | 256 |
| `chronos-t5-small` | 202 | 35% | 1.05 | 4.90 | 0.33 | 256 |
| `timesfm-2.5-200m` | 202 | 29% | 1.06 | 4.89 | 0.36 | 256 |
| `timesfm-3.0` | 202 | 29% | 1.07 | 4.95 | 0.35 | 256 |
| `seasonal_naive` | 202 | 19% | 1.08 | 4.82 | — | 256 |
| `flowstate-r1` | 202 | 20% | 1.10 | 5.01 | 0.21 | 256 |
| `patchtst-fm-r2` | 202 | 20% | 1.14 | 5.37 | — | 256 |
| `moving_average` | 202 | 13% | 1.21 | 5.42 | — | 256 |
| `ttm-r2` | 202 | 27% | 1.25 | 5.70 | — | **512** |
| `timemoe-50m` | 132 | 6% | 1.47 | 6.56 | 0.38 | 256 |
| `moment-1-small` | 132 | 1% | 2.64 | 12.13 | — | 256 (head untrained) |
| `moment-1-large` | 132 | 1% | 2.66 | 12.02 | — | 256 (head untrained) |
| `ttm-r3` | 168 | 1% | 3.75 | 17.60 | — | **512** |

* **On 101 financial series, no foundation model beats a well-tuned ARIMA at h=21/63.**
  ARIMA leads the corpus (55% of cells better than naive, median ratio 1.00); the strongest
  transformers are at parity or slightly worse (Chronos-2 40%, Chronos-Bolt-small 41%,
  TimesFM 29%). This is the headline of the benchmark, and it is a statement about
  *zero-shot* weights on *daily/hourly price data at three-week and quarter horizons*.
* **A green matrix cell and a correct number are different things, twice over.** MOMENT
  scored because we rebuild its task head, but the checkpoint never trained one (flagged,
  `head_from_checkpoint: false`). TTM was scored at ctx 256 while its checkpoint declares
  512 and pads shorter input with zeros, which scaled its whole forecast by the fraction of
  real history and produced a median MASE of 13x naive. Honouring the contract moved r2
  from 1.51 to 1.25 and made r3's real weakness visible instead of hiding it behind the
  artifact.
* **`ttm-r3` is unstable on this corpus, and the harness now says so in the row.** 272 of
  3,232 forecast windows came back non-finite (416 on CPU lanes, 272 on CUDA - so not a
  device artifact), and 34 of 202 cells scored nothing. Those cells report
  `pred_nan_windows` and an `error`, `n_windows` counts only what was scored, and no metric
  column carries a NaN.
* **Model coverage differs across the corpus and the table shows it**: MOMENT and TimeMoE
  ran only on trux's 66 series (no `moment`/`timemoe` env on watson), `chronos-t5-base` has
  33 CUDA cells plus 33 CPU cells after a shared-card OOM. `ctx` and device are per-row
  facts, not footnotes.

* **These numbers are reproducible - across machines and across re-runs.** On unscaled
  metrics the machines agree bit-for-bit: RMSE matched to ~1e-8 and directional accuracy was
  identical when gpu7 (DGX Spark, GB10) re-scored watson's series (`probes/trux/
  cross_machine_check.py`), and 254 repeated cell comparisons on one machine - pass-3 lanes
  at code sha `1d2f3ae` re-scored at `28fd7e7` - agree to within 0.143%. That is the first
  time this benchmark has had a consistency check rather than an assumption.
* **Cross-machine, same data path: bit-exact for ten of eleven families.** gpu7 (GB10, CUDA,
  torch 2.9.1+cu130) vs watson (RTX A6000, CUDA, torch 2.7.1+cu128) over **1,452 shared cells**
  (66 series x 2 horizons x 11 models, same arena parquet bytes): ten families agree at 0.000%
  median and 0.000% max on all 132 cells each; `chronos-t5-small` is the only outlier (median
  5.73%, max 17.4%, 109 of 132 cells). A third machine (trux, x86 CPU) reproduces the same
  numbers bit-exactly for the same families. The exception tracks the *decoding path* (T5 samples
  tokens; nothing else in the fleet does), not the hardware.

* **Scaled metrics are only comparable within a data path.** The 12-series GB10 overlap
  looked like 212 machine disagreements: MASE off by 15%-123% while RMSE agreed to the
  eighth decimal. Cause: watson's pass-3 lane read exported CSVs (so `SeriesDataset`
  synthesized a 70/15/15 split) and the trux/gpu7 lanes read arena directories (the
  publisher's split), and MASE is divided by the naive error of whichever train split was
  loaded. For `std_equity_1d_10104`: 0.375671 vs 0.431936, reciprocal ratio 1.1498 - exactly
  the observed MASE ratio, to four decimals. Rows now carry `data_hash` (values + split
  boundaries) and `data_kind`, the lane meta carries `data_kinds`, the merge files such a
  repeat under supersessions, and the cross-machine check exits "inconclusive" instead of
  "disagreement". `beats naive` per cell survives this (the baseline shares the denominator);
  absolute MASE columns across the watson block (CSV) and the trux block (arena dirs) do not.
* **Repeated cell keys are mostly a protocol being replaced, not noise.** The pass-3 merge
  inputs carry 404 `(model, series, span, horizon)` keys measured more than once. Scored by
  protocol signature (ctx, ctx_note, device, code_sha, torch), they are overwhelmingly
  supersessions - a TTM cell at ctx 256 (the zero-padding artifact, MASE 30.1) and the same
  cell at its contract ctx 512 (MASE 4.3). Reading that spread as run variance would blame
  the noise floor for the bug we removed. The merge now splits the two readings and reports
  each separately; only identical-signature repeats are quoted as variance.
* **A row's `model` field is the adapter's family key, not the registry key**, and ten
  checkpoints collide on it (`chronos-bolt-base` and `-small` both write `chronos-bolt`;
  `kronos-base`/`-mini`/`-small` all write `kronos-small`). Grouping results by that field
  merges different checkpoints and manufactures disagreement — it is exactly how a
  "3-40% cross-machine discrepancy" appeared and disappeared in this session's notes.
  Rows now carry `model_key`; the merge always keyed on the registry key.

What this does **not** license: any variance claim (one seed, 8 windows per cell), any
statement about fine-tuned checkpoints, or cross-context comparison of TTM against the
256-point models without reading the `ctx` column. Reproduce the table with
`probes/trux/rank_pass3.py` over the merged JSONL files.

### The one fine-tune measurement in this benchmark (MOMENT, watson A6000)

The matrix says which families *can* be fine-tuned; only MOMENT has zero-shot and
fine-tuned cells on the same series, because its zero-shot forecast head is randomly
initialised and fine-tuning is the only way it becomes a forecaster. 12 series, ctx 256,
120 fit steps from the train span, scored on the test span
(`reports/moment_ft_watson/lane_ft.jsonl`, read it with
`probes/watson/moment_ft_report.py`). MASE as a ratio to naive, median over the 12:

| adaptation | lr | h=21 | h=63 |
|---|---|---|---|
| zero-shot (head untrained) | - | 2.56 | 2.14 |
| heads_only | 1e-6 | 2.42 | 2.12 |
| heads_only | default | 2.18 | 1.71 |
| heads_only | **1e-4** | **1.67** | **1.55** |
| full fine-tune | default | 1.70 | 1.54 |

- Fine-tuning the head closes about a third of the gap and no more: MOMENT goes from 2.6x
  naive to 1.67x naive and still loses to naive on 11 of 12 series. Some individual cells
  do beat it (best 0.72 of naive at h=63), which is what you would expect on the few
  series with structure a reconstruction-trained encoder can see.
- **Learning rate mattered more than which weights were trainable.** lr 1e-6 barely moved
  the number (2.42, close to zero-shot); lr 1e-4 with only the head trainable matched full
  fine-tuning of the whole model. A fine-tune comparison that leaves lr at a framework
  default can conclude "this family does not adapt" when the truth is "this lr does
  nothing".
- Every fine-tuned row carries `head_from_checkpoint: false, head_trained_here: true`, so
  the table cannot be misread as a pretrained forecast head.

## What passes 4-6 show: 101 series on one data path, with intervals

Pass 4 scored the corpus on a single arena-parquet data path (`gpu7` 66 series + `watson` the other
35, horizons 21/63). Pass 5 had `watson` re-score `gpu7`'s 66 series; pass 6 added the two families
that had never been cross-checked (`moment-1-small`, `moment-1-large`, `timemoe-50m`: `watson` on an
A6000, `trux` on CPU) and scored the five families `gpu7` had skipped (`chronos-t5-base`,
`flowstate-r1`, `timesfm-3.0`, `ttm-r3`, `patchtst-fm-r2`). The merged table is now
**3,628 rows over 101 series from 8 lane files on 4 machines**, and **1,980 of those cells** were
measured twice on two stacks that share nothing (66 series x 2 horizons x 15 model keys). That is what
turns the corpus from a partition into a sample with replication: the merge report classifies all
2,064 repeated pairs as "same bytes, same protocol, different stack", with no protocol changes hiding
inside them.

The ranking lives in `EFFECTIVENESS.md`; what belongs here is the uncertainty around it
(`probes/pass4_share_bootstrap.py`, cluster bootstrap over *series*, each series moving as a block
because its two horizons are not independent):

| model | cells | beat naive | 95% CI |
|---|--:|---|---|
| `arima` | 202 | 55.4% | 48.0 .. 62.9 (101 series) |
| `chronos-bolt-small` | 202 | 41.1% | 34.7 .. 47.5 (101 series) |
| `chronos-2` | 202 | 39.6% | 33.2 .. 45.5 (101 series) |
| `chronos-bolt-base` | 202 | 32.2% | 25.7 .. 39.1 (101 series) |
| `chronos-t5-small` | 202 | 31.7% | 24.8 .. 38.6 (101 series) |
| `chronos-t5-base` | 202 | 30.7% | 24.3 .. 37.1 (101 series) |
| `theta` | 202 | 30.2% | 24.3 .. 36.6 (101 series) |
| `timesfm-2.5-200m` | 202 | 29.2% | 23.3 .. 34.7 (101 series) |
| `timesfm-3.0` | 202 | 29.2% | 23.8 .. 34.7 (101 series) |
| `ttm-r2` | 202 | 26.7% | 20.3 .. 33.7 (101 series) |
| `seasonal_naive` | 202 | 19.3% | 14.4 .. 24.3 (101 series) |
| `flowstate-r1` | 202 | 19.3% | 13.4 .. 25.2 (101 series) |
| `patchtst-fm-r2` | 202 | 18.8% | 13.9 .. 24.3 (101 series) |
| `moving_average` | 202 | 12.9% | 8.4 .. 17.8 (101 series) |
| `timemoe-50m` | 132 | 6.1% | 2.3 .. 9.8 (66 series) |
| `ttm-r3` | 126 | 0.8% | 0.0 .. 2.4 (63 series) |
| `moment-1-small` | 132 | 0.8% | 0.0 .. 2.3 (66 series) |
| `moment-1-large` | 132 | 0.8% | 0.0 .. 2.3 (66 series) |

**These shares are 8-window numbers, and that is a level error, not just noise.** Pass 7 re-scored
the 8-model tie cluster on the same 66 series and node at 32 and 64 windows: every share falls
(`arima` 56.1% -> 40.9% -> 44.7%, `chronos-bolt-small` 42.4% -> 15.2% -> 15.9%, `chronos-2` 39.4% ->
22.0% -> 13.6%), 33.0% of cells change winner between 8 and 64 windows, and 32 is already the
plateau (32w -> 64w moves shares by <= 8 pp while 8w -> 32w moves them by 10-27 pp). Head-to-head on
the same cells: `arima > chronos-bolt-small` 58.3% -> 83.3%, `chronos-2 > chronos-t5-small` 55.3% ->
70.5%, and the "tie" `chronos-bolt-small` vs `chronos-2` resolves *against* bolt-small (53.8% ->
38.6%). Only `chronos-bolt-base` vs `chronos-t5-small` stays a tie at every window count (48.5/50.0/
47.0). The intervals below measure corpus sampling, not window sampling; the two errors are
independent and both are in the number. Pass 7b (`probes/watson/watson_pass7b_rest32.sh` plus a trux lane run) moves the
other 11 model keys to 32 windows so the whole table shares one protocol; until it lands the ranks
below are 8-window ranks, and `probes/pass7_merge_files.txt` is the merge that replaces them (it
deliberately excludes the 8-window lanes - a share at 8 windows is a different measurement, not an
earlier draft of the same one). See docs/08 and `probes/window_sensitivity.py`.

What that says, and what it does not:

* **`arima` over every foundation model is real.** It wins the paired comparison against
  `chronos-bolt-small` in 99.9% of resamples and `chronos-2` in 100.0%. Intervals barely touch.
* **`chronos-bolt-small` vs `chronos-2` is a tie, not 41% > 40%.** Pairwise 61.6% - a mildly
  favoured coin flip. Publishing them as ranks 2 and 3 is over-precision.
* **`chronos-bolt-base` vs `chronos-t5-small` is 53.8%** at 8 windows and stays a tie at 32 and 64
  (50.0%, 47.0% head-to-head): this one is a real tie, which is why it survives the protocol change.

A model whose interval overlaps a neighbour's is tied *as far as this pass can see*. Raising
`--max-windows` narrows these intervals roughly as `1/sqrt(windows)`; scoring more series narrows the
series-level part, which is the part the bootstrap measures.
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
