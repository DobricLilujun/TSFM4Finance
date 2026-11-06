"""T4: hunt the remaining finance datasets with Scrapling (d4vinci/Scrapling).

The rule for this tier: *try hard, then say so honestly*.  Anything that needs an
account, an API key or an institutional licence is recorded in
``data/t4_report.json`` as ``needs_user_provision`` with the exact URL and what is
required, instead of being silently dropped from the benchmark.  Anything that is
genuinely open (public zips, sample archives) is downloaded into ``data/raw/t4/``.

Sources probed:
  open        : UEA/TSB-AD archives, Monash forecast archive mirrors, Duke/SEC TAQ
                samples, LOBSTER sample order books, Databento samples,
                Dukascopy daily bar index, STOOQ bulk index.
  key/licence : Tiingo, Alpha Vantage, Polygon, EODHD, QuantQuote, NYSE TAQ,
                CSMAR/Wind, Kaggle fraud sets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from .common import RAW, record_failure

T4 = RAW / "t4"
T4.mkdir(parents=True, exist_ok=True)
REPORT = RAW.parent / "t4_report.json"

try:                                                    # Scrapling when available
    from scrapling.fetchers import Fetcher, StealthyFetcher       # type: ignore
    HAVE_SCRAPLING = True
except Exception:                                       # noqa: BLE001
    Fetcher = StealthyFetcher = None
    HAVE_SCRAPLING = False


@dataclass
class Source:
    key: str
    what: str
    urls: list[str]
    kind: str            # "file" | "index" | "needs_user_provision"
    requirement: str = ""


SOURCES: list[Source] = [
    Source("uea_classification", "UEA time-series classification archive (external "
           "validity for the classification route)",
           ["http://www.timeseriesclassification.com/Downloads/Archives/"], "index"),
    Source("tsb_ad_anomaly", "TSB-AD anomaly benchmark archive (external validity "
           "for the anomaly route, no price-derived labels)",
           ["http://www.timeseriesclassification.com/Downloads/Archives/",
            "https://zenodo.org/records/7731164"], "index"),
    Source("monash_archive", "Monash Time Series Forecasting Archive (fred_md, "
           "bitcoin, oil_dataset, cif_2016 - the finance-heavy public forecast sets)",
           ["https://forecasters.github.io/time-series-forecasting-archive/",
            "https://github.com/jinkwan1115/TimeSeriesDatasets"], "index"),
    Source("lobster_samples", "LOBSTER limit-order-book sample files (microstructure "
           "tasks: spread/depth features, order-flow imbalance)",
           ["https://lobsterdata.com/info/DataDownloads.php"], "index",
           "free registration (email) on lobsterdata.com"),
    Source("duke_taq", "Duke SEC/NYSE TAQ sample day files (trade/quote microstructure)",
           ["https://taq.cmhg.fuqua.duke.edu/samples/"], "index"),
    Source("databento_samples", "Databento free sample .dbn bundles (top-of-book, "
           "trades for several venues)",
           ["https://databento.com/samples"], "index"),
    Source("dukascopy_index", "Dukascopy historical bar/tick index (FX minute/tick "
           "history; .bi5 needs a decoder)",
           ["https://www.dukascopy.com/swiss/english/marketwatch/historical/"], "index"),
    Source("stooq_bulk", "STOOQ bulk daily CSV index (long-history indices/FX/"
           "commodities used as the Yahoo fallback)",
           ["https://stooq.com/db/h/"], "index"),
    Source("kaggle_fraud", "Kaggle credit-card fraud / transaction anomaly sets",
           ["https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"], "needs_user_provision",
           "Kaggle account + API token"),
    Source("tiingo", "Tiingo long-history, split/dividend-adjusted US equity panels",
           ["https://www.tiingo.com/documentation/end-of-day"], "needs_user_provision",
           "free-tier API key"),
    Source("alphavantage", "Alpha Vantage daily/minute equity + FX panels",
           ["https://www.alphavantage.co/documentation/"], "needs_user_provision",
           "free API key"),
    Source("polygon", "Polygon.io adjusted panels + aggregates (survivorship-aware)",
           ["https://polygon.io/docs"], "needs_user_provision", "paid key"),
    Source("csmar_wind", "CSMAR / Wind A-share panels (the finance anomaly + "
           "cross-sectional data used by the cited A-share studies)",
           ["https://www.gtarsc.com/"], "needs_user_provision",
           "institutional subscription"),
    Source("nyse_taq_full", "Full NYSE TAQ / Databento equities history",
           ["https://www.nyse.com/market-data/equities/us-equity-historical-data"],
           "needs_user_provision", "commercial licence"),
]


def _get(url: str, timeout: int = 60) -> str | None:
    if HAVE_SCRAPLING:
        for fetch in (lambda: Fetcher.get(url, timeout=timeout * 1000),
                      lambda: StealthyFetcher.fetch(url, headless=True, timeout=timeout * 1000)):
            try:
                r = fetch()
                body = getattr(r, "body", None) or getattr(r, "text", "")
                if body:
                    return body
            except Exception:                                # noqa: BLE001
                continue
    try:
        from .common import http_get
        return http_get(url, timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0 tsfm4finance-bench/1.0"}).decode(
            "utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None


def _links(body: str, patterns: list[str]) -> list[str]:
    out = []
    for m in re.finditer(r'href="([^"]+)"', body):
        u = m.group(1)
        if any(p in u.lower() for p in patterns):
            out.append(u)
    return sorted(set(out))


def probe(src: Source) -> dict:
    entry = asdict(src)
    entry["status"] = "unreachable"
    entry["found_links"] = []
    if src.kind == "needs_user_provision":
        entry["status"] = "needs_user_provision"
        return entry
    for url in src.urls:
        body = _get(url)
        if not body:
            continue
        pats = ["zip", "csv", "tar", "parquet", "dbn", "bi5", "download", "sample"]
        links = _links(body, pats)[:12]
        if links:
            entry["status"] = "found_index"
            entry["found_links"] = [url if l.startswith("http") else f"{url.rstrip('/')}/{l.lstrip('/')}"
                                    for l in links]
            break
        entry["status"] = "reachable_no_links"
    return entry


def download_found(src: Source, entry: dict, limit: int = 3) -> dict:
    got = []
    for u in entry.get("found_links", [])[:limit]:
        p = T4 / re.sub(r"[^A-Za-z0-9._-]", "_", u.split("?")[0].split("/")[-1] or "index.html")
        try:
            r = requests.get(u, timeout=300, stream=True,
                             headers={"User-Agent": "Mozilla/5.0 tsfm4finance-bench/1.0"})
            if r.status_code != 200:
                continue
            n = 0
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
                    if n > 400 << 20:                      # 400 MB cap per file
                        break
            if p.stat().st_size > 1000:
                got.append({"url": u, "path": str(p), "bytes": p.stat().st_size})
        except Exception as e:                             # noqa: BLE001
            record_failure("t4", src.key, f"{u}: {e}")
    entry["downloaded"] = got
    if got:
        entry["status"] = "downloaded"
    return entry


def run(force: bool = False, download: bool = True) -> dict:
    rep = {"scrapling": HAVE_SCRAPLING, "sources": {}}
    for src in SOURCES:
        e = probe(src)
        if download and e["status"] == "found_index":
            e = download_found(src, e)
        rep["sources"][src.key] = e
    rep["summary"] = {
        "downloaded": [k for k, v in rep["sources"].items() if v["status"] == "downloaded"],
        "found_not_downloaded": [k for k, v in rep["sources"].items()
                                 if v["status"] in ("found_index", "reachable_no_links")],
        "unreachable": [k for k, v in rep["sources"].items() if v["status"] == "unreachable"],
        "needs_user_provision": [k for k, v in rep["sources"].items()
                                 if v["status"] == "needs_user_provision"],
    }
    REPORT.write_text(json.dumps(rep, indent=1))
    return rep
