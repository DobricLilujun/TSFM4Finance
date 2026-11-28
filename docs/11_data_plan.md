# 11. Data plan — what the Arena has, what it needs, and what to download first

Purpose (from the site): the Arena is an **evaluation venue for other people's TSFMs** — an
**open track** (public data, hidden test truth, anyone can submit) and a **closed track**
(private data, truth never released). That purpose sets three hard requirements that a normal
benchmark does not have:

1. **Statistical resolution** — enough series × enough windows per horizon that a ΔMASE of
   ~0.1 means something (daily data gives 5–8 non-overlapping windows/year at h=21).
2. **Contamination control** — a zero-shot TSFM may already have seen the series during
   pre-training. Public Yahoo daily prices 2004→2024 are almost certainly *inside* those
   corpora (GIFT-Eval literally ships "Bitcoin"), so a "zero-shot win" on them can be memory.
3. **Reproducibility + leak-proofing** — versioned files (sha256 in `data/manifest.json`),
   fixed splits, and for the closed track the test truth never leaves the server.

---

## 1. What we have right now (measured on disk, `data/open`, 2026-11-11)

| bucket | count | what it is | verdict |
|---|---|---|---|
| `core12` | 12 | daily, **close only**, 753 obs (2023-08→2026-08/11) | resolution-poor; 1 regime |
| `long12` | 12 | daily **OHLCV+volume**, 2 298–5 941 obs (2004/2014→2026-11-06) | good spine, but 100 % Yahoo and pre-2025 is probably in pre-training corpora |
| `newx` (T2) | 32 | VIX, yields/spreads, FX, commodities, global indices, sector ETFs | adds asset-class breadth (downloaded on the cluster) |
| `fred` | 31 ids coded | macro/rates/credit | **0 on disk**: `fred.stlouisfed.org` unreachable from the cluster (29/31 timeouts) |
| `intraday` | **0** | planned Binance bars | the single biggest missing resolution lever |
| `external` | **0** | GIFT-Eval / fev-bench anchors | needed for §4.3 of the paper and for outside comparability |
| derived tasks | 96 | `_dir5`, `_volreg`, `_extreme`, `_trade` (24 series × 4) | labels come from **our own rules** → weak for the anomaly track |
| base total | 60 | sources: **yahoo 45** + legacy stubs; 3 of the legacy ones are **synthetic fallbacks** | single-vendor risk; synthetics must be excluded from the Arena |
| `data/closed/priv_*` | 4 stubs | `priv_broker_order`, `priv_credit_risk`, `priv_crypto_tick`, `priv_fund_flow` (empty parquet) | the closed track does not exist yet — needs your data |

Crypto today: `btc_long` (OHLCV 4 433 bars, 2014→), `eth_long`, `btc_crypto`, `eth_crypto`
(close-only) — **2 assets**, both from Yahoo. Against a survey section (§5.3.4) that discusses
crypto at length, that is the thinnest part of the whole dataset.

## 2. Contamination policy (this decides what is worth downloading)

A series is treated as **clean** only if at least one holds:

* **C-fresh** — its evaluation window starts after the plausible pre-training cut-off. Today we
  can use **2026-08 → now** as a genuinely unseen slice for every series we refresh.
* **C-new** — the instrument was *listed* after the corpus cut-off (tokens listed 2025-01 or
  later): its entire history post-dates the corpora.
* **C-rare** — the series type is rarely published in TS corpora: **funding rates, open
  interest, liquidations, options implied volatility, order-book/trade-flow, on-chain flows**.
* **C-private** — closed track only.

Every downloaded dataset gets meta fields `listed_at`, `first_seen`, `source_url`, `license`,
`contamination` (`fresh` / `post_listing` / `rare` / `public_risky`) and `fresh_split`
(first date of the post-cut-off window), so a submission can be scored on the clean slice only.

## 3. Download list (priority = "hard to get × clean × cheap to wire")

| id | data | endpoint (no key unless noted) | size | cleanliness | status |
|---|---|---|---|---|---|
| **N1** | Binance **spot daily** for ~40 pairs, 2019→now (OHLCV+quote-volume+trades+taker-buy) | `data.binance.vision/data/spot/monthly/klines/{PAIR}/1d/…zip` (probed 200 OK) | ~50 MB | `public_risky`, but `post_listing` for new listings | ⬜ |
| **N2** | Binance spot **1h**, 2019→now, 12 big pairs | same, `/1h/` | ~1.5 GB | `public_risky` but huge window count | ⬜ |
| **N3** | Binance spot **5m/1m last ~120 days**, 10 pairs | `data/spot/daily/klines/{PAIR}/5m/…` | ~500 MB | **`fresh`** (post-cut-off) | ⬜ |
| **N4** | Binance **USDT-M futures**: klines 1h + **funding rate** + **open interest** (+ liquidation where still published) | `data/futures/um/monthly/{klines,fundingRate,openInterest}/…` | ~300 MB | **`rare`** — derivatives flows are not in TS corpora | ⬜ |
| **N5** | **Newly listed** tokens (listing date ≥ 2025-01 from `exchangeInfo`/first-candle scan) → their full daily history | Binance | ~50 MB | **`post_listing`** = clean by construction | ⬜ |
| **N6** | **Stablecoin pegs**: USDCUSDT, DAIUSDT, FDUSDUSDT, BUSDUSDT (depeg events as labelled anomalies) | Binance daily/hourly | ~20 MB | `rare` + gives *externally labelled* anomaly truth | ⬜ |
| **N7** | **Deribit DVOL / BTC-ETH implied vol** | `api.deribit.com/api/v2/public/plot_volatility?index=BTC` (deribit.org unreachable; retry api host) | ~5 MB | `rare` | ❓ reachability |
| **N8** | FRED macro panel (31) | `fredgraph.csv` — works **from this laptop**, not from the cluster | ~10 MB | `public_risky` | ⬜ then rsync to the cluster |
| **N9** | GIFT-Eval econ subset + fev-bench finance tasks | `hf download Salesforce/GiftEval --repo-type=dataset`; `fev` | 1–3 GB | anchors (not clean) | ⬜ |
| **N10** | closed track | your broker flow / credit / crypto ticks / fund flow | ❓ | `private` | ❓ your upload |

Skip on purpose: LOBSTER/FI-2010 (institutional/request-gated — revisit if you have WRDS),
CRSP/Compustat/OptionMetrics (same), Twitter/news archives (needs keys + licence review).

## 4. Wiring after download (so the data is actually used, not just stored)

1. `bench/download/binance.py` extended: pair list (40), futures (`fundingRate`,
   `openInterest`, UM klines), daily-file mode for 5m/1m, `exchangeInfo` listing dates.
2. `bench/download/deribit.py` (DVOL + IV) and `fred.py` rerun from this machine.
3. New registry groups in `bench/config.yaml`: `crypto40`, `crypto_deriv`, `crypto_hi`,
   `fred`, `external` — empty groups are invisible to the sweep, so this is the step that
   makes the new data enter every stage.
4. `--stage derive` on the new groups → `_dir5` (direction), `_volreg` (regime), `_extreme`
   (anomaly), `_trade` (cost-aware trading) for every new series; for stablecoins also a
   **depeg event** truth built from the *external* record, not our MAD rule.
5. Contamination flags into `meta.json` + `data/manifest.json`; `bench/protocol.py` gains a
   `span="fresh"` so any model can be scored on the post-cut-off window only.
6. Arena closed track: mark each dataset `open|closed` in meta, keep `test.parquet` off the
   public download (server-side truth for `POST /api/submit`), and document the submission
   contract in this file.

## 5. Acceptance checks (run after the downloads, before we call it done)

* per dataset: `rows ≥ 1 800` (≥ 8 windows at h=21 for 1h data: ≥ 300), columns present,
  no duplicate timestamps, gap audit (crypto must be 24/7 — gaps are a bug there),
  `date_range` recorded, sha256 in manifest;
* per group: printed counts in `--stage status`;
* a smoke forecast cell per new group (proves the parquet is loadable by every stage);
* leakage check: `no_leakage=true` on all rows **and** `fresh` windows never overlap `train`.

## 6. Rough cost

* N1+N3+N4+N5 with 6–8 parallel workers: ~30–60 min wall-clock, ~1 GB.
* N2 (1h × 12 pairs): ~1 h, 1.5 GB.
* N8 FRED from this laptop: ~10 min. N9: 1–3 GB, ~1 h.
* Disk: we have room locally (and 3.5 TB free on the cluster home).

## 7. Decisions I still need from you

1. **How many crypto pairs** for the open track — 20 (fast, clean) or 50 (better breadth, more
   dead/renamed tickers to filter)? Default chosen: **40 majors + all tokens first listed 2025+**.
2. **Closed track**: can you provide any of `priv_broker_order / priv_credit_risk /
   priv_crypto_tick / priv_fund_flow`? Without them the closed half of the site is a design
   statement, not evidence.
3. Should the public download exclude `test.parquet` (making the open track actually
   hidden-truth like the closed one)? Default: **yes** — it is the whole point of the site.
