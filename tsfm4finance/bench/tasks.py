"""Build **real-data** task datasets from the arena's 12 real price series (spec §2.1-§2.3).

Everything here is derived from actual market closes (`data/open/<base>/`) — no synthetic
labels. Four task views per series:

  *_dir5     task=classify  label = sign of the next h-day log return (h=5).
             Decision at t (prices <= t); the label is the target, never a feature.
  *_volreg   task=classify  label = tercile (calm/normal/turbulent) of the realized vol
             of the next h days; tercile cut-points are estimated on the TRAIN split only.
  *_extreme  task=anomaly   label = 1 when the return AT time t exceeds an expanding
             median +/- 3*MAD threshold estimated from data strictly before t
             (real extreme-move labels, no injection, no look-ahead).
  *_trade    task=trading   same series, trading fields (horizon, cost_bps) per spec §2.3.

Splits mirror the source dataset exactly (same chronological train/val/test boundary),
so every task is scored on the same held-out period.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsfm4finance.paths import DATA
from tsfm4finance.core.schemas import DatasetMeta, Domain, Frequency, TaskType

HERE = Path(__file__).resolve().parent

# the 12 real (yfinance) series; synthetic ones are excluded by decision Q1
REAL_SERIES: list[tuple[str, str]] = [
    ("sp500_daily", "index"),
    ("nasdaq_daily", "index"),
    ("dow_daily", "index"),
    ("russell2000_daily", "index"),
    ("btc_crypto", "crypto"),
    ("eth_crypto", "crypto"),
    ("eur_usd_fx", "fx"),
    ("gbp_usd_fx", "fx"),
    ("gold_commodity", "commodity"),
    ("crude_oil_commodity", "commodity"),
    ("us_treasury_7y", "bond"),
    ("ig_corp_bond", "bond"),
]

_DIR_H = 5          # direction horizon: one business week
_VOLREG_H = 21      # volatility-regime horizon: one business month
_EXTREME_Z = 3.0    # extreme-move threshold in expanded-MAD units


def _source_dir(name: str) -> Path:
    return DATA / "open" / name


def _load_source(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = _source_dir(name)
    return (pd.read_parquet(d / "train.parquet"),
            pd.read_parquet(d / "validation.parquet"),
            pd.read_parquet(d / "test.parquet"))


_FREQ_MAP = {"tick": Frequency.TICK, "second": Frequency.SECOND, "1m": Frequency.MINUTE,
             "5m": Frequency.MINUTE, "15m": Frequency.MINUTE, "30m": Frequency.MINUTE,
             "minute": Frequency.MINUTE, "1h": Frequency.HOUR, "8h": Frequency.HOUR,
             "hour": Frequency.HOUR, "1d": Frequency.DAY, "day": Frequency.DAY,
             "daily": Frequency.DAY, "weekly": Frequency.WEEK, "week": Frequency.WEEK,
             "monthly": Frequency.MONTH, "month": Frequency.MONTH,
             "quarterly": Frequency.QUARTER, "quarter": Frequency.QUARTER}


def _source_profile(name: str) -> tuple[str, Frequency]:
    """The source dataset's target column and its frequency.

    Crypto derivatives series target ``funding_rate``, macro series target ``close`` (a
    level).  The task views always write the modelled column as ``close`` because that is
    what the harness scores, so a non-close target is renamed and the note says so.
    """
    import json
    meta_path = _source_dir(name) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    freq = _FREQ_MAP.get(str(meta.get("frequency", "")).lower(), Frequency.DAY)
    return str(meta.get("target") or "close"), freq


def _is_price(close: np.ndarray) -> bool:
    """True when a log return is defined, i.e. every observation is strictly positive.

    Funding rates, spreads and OAS levels cross zero, so "return" has to mean the first
    difference there - otherwise np.log produces NaNs and the labels silently degrade.
    """
    c = close[~np.isnan(close)]
    return bool(c.size) and float(np.min(c)) > 0


def _logret(close: np.ndarray) -> np.ndarray:
    if not _is_price(close):                     # rates / spreads: level change
        r = np.empty_like(close, dtype=float)
        r[0] = 0.0
        r[1:] = close[1:] - close[:-1]
        return r
    r = np.empty_like(close, dtype=float)
    r[0] = 0.0
    r[1:] = np.log(close[1:] / close[:-1])
    return r


def _fwd_ret(close: np.ndarray, h: int) -> np.ndarray:
    """forward change from t to t+h: log return for prices, level change for rates."""
    out = np.full(len(close), np.nan)
    if not _is_price(close):
        out[:-h] = close[h:] - close[:-h]
        return out
    out[:-h] = np.log(close[h:] / close[:-h])
    return out


def _fwd_vol(close: np.ndarray, h: int) -> np.ndarray:
    """realized vol of the h daily returns inside (t, t+h] (NaN near the end)."""
    r = _logret(close)
    out = np.full(len(close), np.nan)
    for t in range(len(close) - h):
        out[t] = float(np.std(r[t + 1:t + 1 + h], ddof=1))
    return out


def _extreme_labels(close: np.ndarray, z: float = _EXTREME_Z) -> np.ndarray:
    """Expanding-window MAD flag on the return AT time t (no look-ahead)."""
    r = _logret(close)
    lab = np.zeros(len(close))
    for t in range(1, len(close)):
        hist = r[:t]                       # strictly before t
        if len(hist) < 30:
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        if mad <= 0:
            continue
        if abs(r[t] - med) > z * 1.4826 * mad:
            lab[t] = 1.0
    return lab


def _write_dataset(name: str, df_all: pd.DataFrame, n_train: int, n_val: int,
                   meta: DatasetMeta) -> Path:
    d = DATA / "open" / name
    d.mkdir(parents=True, exist_ok=True)
    tr = df_all.iloc[:n_train]
    va = df_all.iloc[n_train:n_train + n_val]
    te = df_all.iloc[n_train + n_val:]
    tr.to_parquet(d / "train.parquet")
    va.to_parquet(d / "validation.parquet")
    te.to_parquet(d / "test.parquet")
    meta.n_obs = int(len(df_all))
    if "timestamp" in df_all.columns:
        ts = pd.to_datetime(df_all["timestamp"])
        meta.date_range = [str(ts.min().date()), str(ts.max().date())]
    meta.feature_names = list(df_all.columns)
    meta.n_features = int(len(df_all.columns))
    (d / "meta.json").write_text(meta.model_dump_json(indent=2))
    return d


def build_for_series(base: str, domain: str) -> list[str]:
    """Create the 4 task datasets for one real series. Returns the new dataset names."""
    tr, va, te = _load_source(base)
    full = pd.concat([tr, va, te], ignore_index=True)
    target, freq = _source_profile(base)
    renamed = ""
    if target != "close" and target in full.columns:
        full = full.rename(columns={target: "close"})
        renamed = f" modelled column is the source target {target!r} renamed to close;"
    close = full["close"].to_numpy(dtype=float)
    n_train, n_val = len(tr), len(va)
    src = full["timestamp"] if "timestamp" in full.columns else None
    created: list[str] = []

    # ---- 1) direction classification (next 5-day return sign) ----
    fr = _fwd_ret(close, _DIR_H)
    dir_df = pd.DataFrame({"timestamp": src, "close": close,
                           "label": np.where(np.isnan(fr), np.nan, (fr > 0).astype(float))})
    m = DatasetMeta(name=f"{base}_dir5", domain=Domain(domain), task=TaskType.CLASSIFY,
                    frequency=freq, n_features=3, feature_names=[],
                    source=f"{base} (real) + derived label", open=True, target="close",
                    lookback=512, horizon=_DIR_H, n_classes=2, labels=["down", "up"],
                    notes=(f"sign of the {_DIR_H}-bar forward "
                           f"{'log return' if _is_price(close) else 'change'}{renamed}; "
                           f"label is the target (uses the future), "
                           "features use prices <= t"))
    created.append(_write_dataset(f"{base}_dir5", dir_df, n_train, n_val, m).name)

    # ---- 2) volatility-regime classification (next 21-day realized vol tercile) ----
    fv = _fwd_vol(close, _VOLREG_H)
    train_fv = fv[:n_train]
    cuts = np.nanquantile(train_fv, [1 / 3, 2 / 3])
    reg = np.full(len(close), np.nan)
    ok = ~np.isnan(fv)
    reg[ok] = np.digitize(fv[ok], cuts)                # 0 calm, 1 normal, 2 turbulent
    reg[np.isnan(fv)] = np.nan
    vol_df = pd.DataFrame({"timestamp": src, "close": close, "label": reg})
    name = f"{base}_volreg"
    m = DatasetMeta(name=name, domain=Domain(domain), task=TaskType.CLASSIFY,
                    frequency=freq, n_features=3, feature_names=[],
                    source=f"{base} (real) + derived label", open=True, target="close",
                    lookback=512, horizon=_VOLREG_H, n_classes=3,
                    labels=["calm", "normal", "turbulent"],
                    notes=f"tercile of realized vol over the next {_VOLREG_H} business days; "
                          f"cut-points fitted on the TRAIN split only ({np.round(cuts,5).tolist()})")
    _write_dataset(name, vol_df, n_train, n_val, m)
    created.append(name)

    # ---- 3) extreme-move anomaly (real labels, expanding MAD, no look-ahead) ----
    ext = _extreme_labels(close)
    ext_df = pd.DataFrame({"timestamp": src, "close": close, "label": ext})
    name = f"{base}_extreme"
    m = DatasetMeta(name=name, domain=Domain(domain), task=TaskType.ANOMALY,
                    frequency=freq, n_features=3, feature_names=[],
                    source=f"{base} (real) + derived label", open=True, target="close",
                    lookback=512, horizon=1, n_classes=2, labels=["normal", "extreme_move"],
                    notes=f"|daily return| beyond expanding median +/- {_EXTREME_Z} MAD "
                          f"(threshold from data strictly before t); base rate "
                          f"{ext.mean():.3f}")
    _write_dataset(name, ext_df, n_train, n_val, m)
    created.append(name)

    # ---- 4) trading view (spec §2.3) ----
    trd_df = pd.DataFrame({"timestamp": src, "close": close})
    name = f"{base}_trade"
    m = DatasetMeta(name=name, domain=Domain(domain), task=TaskType.TRADING,
                    frequency=freq, n_features=2, feature_names=[],
                    source=f"{base} (real)", open=True, target="close",
                    lookback=512, horizon=_DIR_H, cost_bps=10.0,
                    labels=["long", "flat", "short"],
                    notes="signal -> position -> cost-aware PnL (spec §2.3); "
                          "positions are held over non-overlapping h-day intervals")
    _write_dataset(name, trd_df, n_train, n_val, m)
    created.append(name)

    return created


def build_all() -> list[str]:
    made: list[str] = []
    for base, domain in REAL_SERIES:
        made += build_for_series(base, domain)
    return made


if __name__ == "__main__":
    names = build_all()
    print(f"created {len(names)} datasets:")
    for n in names:
        print("  ", n)
