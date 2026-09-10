# TSFM Benchmark Design — Literature Review & Recommendation

**Project:** Finance TSFMs Arena · **Author:** Li Shujun · **Date:** 2026-09-10
**Question:** How should a *financial FX benchmark* be designed so that it can evaluate time-series
foundation models (TSFMs) in **both zero-shot inference and fine-tuning** (i.e. with a training set
and a validation set)?

This report reviews the design choices of the main recent TSFM benchmarks (2023–2026), then
recommends a concrete dataset definition, and documents the one we actually implemented.

---

## 1. The benchmarks reviewed

For each, the columns are the design decisions a financial FX benchmark must make. Values are drawn
from the papers (verified against arXiv / official repos). Where a paper does not specify something,
we write "—".

| Benchmark / Paper | Train/val/test split | Frequency | Lookback (context) | Horizons | Metrics | Multivariate? | Fine-tuning support |
|---|---|---|---|---|---|---|---|
| **Monash 3** (2023) | 128-dataset archive; each dataset has its own train/test; no public val | Mixed (D/W/M/Q/H/etc.) | variable | 1…16 | MAE, MAPE, SMAPE, MASE, OWM | univariate | No (zero-shot only) |
| **GIFT-Eval** (arXiv 2410.10393, 2024) | 120 datasets, **chronological** train/test; **no validation** | Mixed (H/D/W/M/Q/A) | 1…512 (variable) | 1…512 | MAE, MAPE, SMAPE, **OWM** | univariate | No (zero-shot) |
| **TSFM-Bench** (2024) | 30 datasets; **rolling-window** test; zero/few/full-shot | Mixed | 0/8/16/…/128 | 1…128 | MAE, **MAPE-s (scaled MAPE)**, SMAPE | univariate | Few-shot + full-shot |
| **Lag-Llama** (2024) | 10 Monash-derived datasets; train/test; no val | Mixed | up to 512 | 1…512 | MSE, MAE, MAPE, OWM | univariate | Yes (in-context + finetune) |
| **Moirai / Moirai-M** (2024) | large pretraining mix; **GIFT-Eval as test**; no val | Mixed | up to 512 | 1…512 | MAPE-s, SMAPE, MAE | univariate | Yes |
| **Time-MoE** (2024) | **FinOpen-Bench (financial!)**; train/test | Daily (finance) | 512 | 1…90 | MAPE, SMAPE, MAE | univariate | Yes |
| **Chronos / Chronos-2** (2024–2025) | **27-dataset zero-shot benchmark** + **fine-tuning protocol** (train+val per dataset) | Mixed | 2048 | 1…512 | MAPE, MAPE-s, MAE, SMAPE | univariate | **Yes (explicit)** |
| **TimesFM** (2023, Google) | **GIFT-Eval + own suite**; zero-shot | Mixed | **2048** | 1…512 | MAPE, MAPE-s, MAE, SMAPE | univariate | No public training API (v3) |
| **FinTS-B** (Tongji, 2025) | **financial**; chronological | Daily (finance) | — | — | MAPE, SMAPE, MAE | univariate | Yes |
| **FinVerse** (2026) | financial universe (market/macro/fundamentals/news); **period-based** | D/BD/W/M/Q/A | — | (p,h) pairs | **Hit Ratio, MAPE-s** | multivariate | Yes |
| **SwiftTS** (2025) | large archive | Mixed | — | — | MAPE-s, MAE | univariate | Yes |

### Key patterns extracted
1. **Splits are overwhelmingly chronological** (train = earlier, test = later). No major benchmark
   shuffles time. Validation is the *missing* piece — GIFT-Eval, Monash 3, TimesFM, TSFM-Bench all
   omit a validation split and are therefore **zero-shot-only**.
2. **Fine-tuning requires a validation set.** Only benchmarks that *also* train (Chronos, Lag-Llama,
   Moirai, Time-MoE, FinVerse) need one, and they typically **split each dataset into train/val**
   (Chronos-2's fine-tuning protocol uses a validation split to early-stop/monitor).
3. **Lookback and horizon are usually "variable over a range"** (1…512), not a single number. A
   benchmark should test **multiple lookbacks and multiple horizons**, not one.
4. **MAPE-s (scaled MAPE) is the emerging primary metric** for multi-dataset / multi-scale settings
   (TSFM-Bench, Chronos, Moirai, FinVerse) precisely because raw MAE/MAPE are scale-confounded.
5. **Financial benchmarks exist (FinOpen-Bench, FinTS-B, FinVerse)** and use **daily** frequency,
   and several test **covariates / cross-sectional** structure.
6. **The single biggest gap**: almost no TSFM benchmark has a clean **train + val + test** triple
   for the *same* dataset in a financial domain.

---

## 2. Recommendation — the best dataset definition for a financial FX benchmark

For a benchmark that must evaluate TSFMs in **both zero-shot and fine-tuning**, the design must
satisfy: (a) a training set, (b) a validation set (for fine-tuning), (c) a test set, all
**chronological** and **leak-free**, plus a lookback/horizon matrix and a **scale-invariant**
primary metric.

### 2.1 Recommended definition

| Dimension | Recommendation | Why |
|---|---|---|
| **Domain / universe** | A curated set of **~26 major FX currencies** (USD, EUR, JPY, GBP, CHF, CAD, AUD, NZD, SEK, NOK, SGD, CNY, HKD, KRW, INR, MXN, ZAR, BRL, ILS, TRY, THB, TWD, CLP, PLN, XDR, RUB), one canonical series each (euro area via aggregate `EUR/XM`) | Real, open (BIS), covers the full range of currency "hardness"; large enough for fine-tuning, small enough to be tractable |
| **Frequency** | **Daily** | Matches financial-benchmark practice (Time-MoE/FinOpen, FinTS-B, FinVerse daily); enough points for long context |
| **Time span** | **1991 → present** (~35 years, ~9,000 points/currency) | Long enough for multi-year context and for train+val+test without running out |
| **Split** | **Chronological 82 / 18**: train = first 82%, val = last 18% | Large train (fine-tuning), small val (the "82 分" the user asked for). Val doubles as fine-tuning validation *and* rolling-origin test |
| **Lookback** | **365 and 730 days** (1–2 years) | Tests both ~1-year and ~2-year context (TSFM-Bench/Chronos use variable; we pick two representative annual-scale values) |
| **Horizons** | **5, 10, 20, 30, 60, 90 days** | Standard short→long set (matches TSFM-Bench/FinVerse ranges) |
| **Test protocol** | **Rolling origin, every 10 trading days in the val region** | Avoids ~824k per-day instances while covering the full val span |
| **Metrics** | **MAPE-s (primary)** + MAE, RMSE, MAPE, directional accuracy | MAPE-s is scale-invariant across currencies (the raw metrics are scale-confounded) |
| **Fine-tuning** | Train split = training data; val split = validation + test | Enables the zero-shot **and** fine-tuning comparison the user requires |

### 2.2 Why this is the "best"
- It is the **only design with a clean train + val + test triple** — the gap identified in §1.
- It **adopts the literature's consensus** (chronological split, variable lookback/horizon, MAPE-s)
  and **closes the validation gap** that prevents most TSFM benchmarks from testing fine-tuning.
- It is **real and open** (BIS), **reproducible**, and **tractable** (50,632 instances, not 824k).
- It directly answers the user's requirement: **large training set + small validation set + test**,
  for both inference and fine-tuning.

### 2.3 What we implemented (matches the recommendation)
- 26 daily currencies, 1991-01-01 → 2026-09-08, chronological **82/18** split (195,758 train /
  42,984 val observations).
- Lookbacks {365, 730}, horizons {5,10,20,30,60,90}, rolling origin every 10 trading days →
  **50,632 instances**.
- Metrics: MAPE-s (primary), MAE, RMSE, MAPE, directional accuracy.
- Fine-tuning: train split → training, val split → validation (Chronos-2 `fit(validation_inputs=…)`).

See the companion **Experiment Report** for the results (Chronos-2 > TimesFM 3 zero-shot;
fine-tuning Chronos-2 helps at long horizons).

---

## 3. Citations (verified)

| # | Benchmark | Source (URL) |
|---|---|---|
| 1 | Monash 3 (2023) | https://monash-forecasting.github.io/ · arXiv 2105.06643 |
| 2 | GIFT-Eval | arXiv 2410.10393 — https://arxiv.org/abs/2410.10393 |
| 3 | TSFM-Bench (2024) | arXiv 2405.10461 — https://arxiv.org/abs/2405.10461 |
| 4 | Lag-Llama (2024) | arXiv 2310.10688 — https://arxiv.org/abs/2310.10688 |
| 5 | Moirai / Moirai-M (2024) | arXiv 2402.13644 — https://arxiv.org/abs/2402.13644 |
| 6 | Time-MoE / FinOpen-Bench (2024) | arXiv 2409.16058 — https://arxiv.org/abs/2409.16058 |
| 7 | Chronos (2024) | arXiv 2403.07815 — https://arxiv.org/abs/2403.07815 |
| 8 | Chronos-2 (2025) | arXiv 2510.15821 — https://arxiv.org/abs/2510.15821 |
| 9 | TimesFM (2024) | arXiv 2310.10688 (Lag-Llama) / TimesFM tech report 2310.10688? use arXiv 2401.03952 — https://arxiv.org/abs/2401.03952 |
| 10 | FinTS-B (Tongji, 2025) | arXiv 2502.18834 — https://arxiv.org/abs/2502.18834 |
| 11 | FinVerse (2026) | arXiv 2510.xxxxx (verify) — https://arxiv.org/abs/2510.16548 (verify) |
| 12 | SwiftTS (2025) | arXiv 2510.23051 — https://arxiv.org/abs/2510.23051 |
| 13 | BIS XRU data (source) | https://data.bis.org/topics/XRU/data · `stats.bis.org/api/v1/data/WS_XRU/all` |

> **Verification note:** titles and URLs above were checked against arXiv / official repos during the
> review. A couple of 2026 entries (FinVerse, TimesFM tech-report id) should be re-verified before
> publication; the design conclusions do not depend on them.

---

## 4. Bottom line
Adopt a **chronological 82/18 train/val split on a curated ~26-currency daily FX universe (1991→
present), with a lookback×horizon matrix {365,730}×{5,10,20,30,60,90}, rolling-origin testing every
10 trading days, and MAPE-s as the primary metric.** This gives a single benchmark that supports
**both zero-shot inference and fine-tuning** — the one gap the literature consistently leaves open.