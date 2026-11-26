"""T1 macro/rates/credit panel from FRED (no API key: fredgraph.csv).

Note: FRED retires ids silently, so the list below is verified against the live endpoint
(2026-11); a retired id now fails loudly instead of producing a bogus one-column frame.

Why macro belongs in a finance TSFM benchmark: macro series are exactly where
TSFM papers claim zero-shot strength (monthly/quarterly, long history, many
related variates), while the finance-market series are where they fail.  Running
both families through the identical rolling protocol is what turns "TSFMs work on
Econ/Fin data" into a precise claim about *which* financial data.
"""
from __future__ import annotations

import io

import pandas as pd

from .common import (dataset_exists, record_failure, clear_failure, http_get,
                     retry, write_open_dataset)

# (fred id, dataset name, domain, frequency label)
SERIES: list[tuple[str, str, str, str]] = [
    ("DGS2", "fred_dgs2", "rates", "daily"),
    ("DGS5", "fred_dgs5", "rates", "daily"),
    ("DGS10", "fred_dgs10", "rates", "daily"),
    ("DGS30", "fred_dgs30", "rates", "daily"),
    ("T10Y2Y", "fred_10y2y_spread", "rates", "daily"),
    ("T10Y3M", "fred_10y3m_spread", "rates", "daily"),
    ("T10YIE", "fred_10y_breakeven", "inflation", "daily"),
    ("DFII10", "fred_10y_tips_real", "inflation", "daily"),
    ("DFF", "fred_fed_funds_daily", "rates", "daily"),
    ("EFFR", "fred_effr", "rates", "daily"),
    ("SOFR", "fred_sofr", "rates", "daily"),
    ("BAMLH0A0HYM2", "fred_hy_oas", "credit", "daily"),
    ("BAMLC0A0CM", "fred_ig_oas", "credit", "daily"),
    ("VIXCLS", "fred_vix", "volatility", "daily"),
    ("NFCI", "fred_nfcI", "financial_conditions", "weekly"),
    ("STLFSI4", "fred_stlfsi", "financial_conditions", "weekly"),
    ("BAMLH0A0HYM2", "fred_high_yield_oas", "credit", "daily"),
    ("UNRATE", "fred_unemployment", "macro", "monthly"),
    ("CPIAUCSL", "fred_cpi", "inflation", "monthly"),
    ("PCEPI", "fred_pce_price", "inflation", "monthly"),
    ("M2SL", "fred_m2", "macro", "monthly"),
    ("INDPRO", "fred_industrial_production", "macro", "monthly"),
    ("HOUST", "fred_housing_starts", "macro", "monthly"),
    ("HSN1F", "fred_house_price_index", "macro", "monthly"),
    ("DTWEXBGS", "fred_usd_broad_index", "fx", "daily"),
    ("TOTCI", "fred_ci_loans", "credit", "weekly"),
    ("WALCL", "fred_fed_balance_sheet", "macro", "weekly"),
    ("DTWEXBGS", "fred_dollar_broad_index", "fx", "daily"),
    ("NFCI", "fred_financial_conditions", "macro", "weekly"),
    ("GDP", "fred_gdp_nominal", "macro", "quarterly"),
    ("GDPC1", "fred_gdp_real", "macro", "quarterly"),
]


@retry(times=3, base_delay=2.0)
def _fred_csv(fid: str, start: str = "2004-01-01", min_obs: int = 100) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}&cosd={start}"
    body = http_get(url, timeout=90).decode("utf-8", "replace")
    if body.lstrip().startswith(("<", "{")):
        raise RuntimeError(f"FRED id {fid!r} is not available (HTML/JSON error page) - "
                           "FRED retires series ids without notice")
    df = pd.read_csv(io.StringIO(body))
    if df.shape[1] < 2:
        raise RuntimeError(f"unexpected FRED response for {fid}: {df.columns.tolist()}")
    date_col, val_col = df.columns[0], df.columns[1]
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[date_col], errors="coerce"),
                        "close": pd.to_numeric(df[val_col], errors="coerce")})
    out = out.dropna().reset_index(drop=True)
    if len(out) < min_obs:      # quarterly series legitimately have ~90 rows since 2004
        raise RuntimeError(f"too few observations for {fid}: {len(out)}")
    return out


_MIN_OBS = {"quarterly": 40, "monthly": 90, "weekly": 150, "daily": 400}


def fetch_one(fid: str, name: str, domain: str, freq: str, *,
              force: bool = False, start: str = "2004-01-01") -> str:
    if dataset_exists(name) and not force:
        return "exists"
    try:
        # a quarterly series since 2004 has ~90 rows, which is plenty for h<=12 forecasts
        df = _fred_csv(fid, start, min_obs=_MIN_OBS.get(freq, 100))
    except Exception as e:                                   # noqa: BLE001
        record_failure("fred", name, str(e))
        return f"failed:{e}"
    write_open_dataset(
        name, df, freq=freq, domain=domain, source=f"fred:{fid}",
        license_="public-domain (FRED)", target="close",
        notes="FRED series via fredgraph.csv; levels as published (some are "
              "percentages/indices - MASE scale is computed per series)",
        extra_meta={"fred_id": fid, "ohlcv": False},
    )
    clear_failure("fred", name)
    return "ok"


def run(force: bool = False) -> dict:
    return {name: fetch_one(fid, name, dom, fq, force=force)
            for fid, name, dom, fq in SERIES}
