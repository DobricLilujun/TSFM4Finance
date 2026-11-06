"""Tiered prefetch: `python -m tsfm4finance.bench.download --tier t1,t2,t3,t4`.

Tiers follow the plan file: t1 = long-history OHLCV market panel + FRED macro,
t2 = cross-sectional panel + external benchmarks (fev-bench / GIFT-Eval),
t3 = intraday crypto bars, t4 = Scrapling hunt for the remaining sources with an
honest report of what needs the user's credentials or licence.
"""
from __future__ import annotations

import argparse
import json

from .common import FAILURES, MANIFEST, manifest_read

# imported inside prefetch(): the external benchmark tier pulls in ``datasets``
# and the scraper pulls in Scrapling - optional heavy deps that must not be
# required by ``from ...download.common import load_dataset``.
binance = external = fred = t4_scrape = yf_panel = None


def prefetch(tiers: tuple[str, ...] = ("t1", "t2", "t3", "t4"), *, force: bool = False,
             xs_limit: int = 0, intraday_interval: str = "1h",
             intraday_start: str = "2019-01", download_t4: bool = True) -> dict:
    global binance, external, fred, t4_scrape, yf_panel
    if "t1" in tiers and yf_panel is None:
        from . import fred as _f, yf_panel as _y
        fred, yf_panel = _f, _y
    if ("t2" in tiers and external is None) or ("t3" in tiers and binance is None):
        from . import binance as _b, external as _e
        binance, external = _b, _e
    if "t4" in tiers and t4_scrape is None:
        from . import t4_scrape as _t
        t4_scrape = _t
    out: dict[str, object] = {}
    if "t1" in tiers:
        out["t1.yf_long12"] = yf_panel.run_long12(force=force)
        out["t1.yf_newx"] = yf_panel.run_newx(force=force)
        out["t1.fred"] = fred.run(force=force)
    if "t2" in tiers:
        out["t2.cross_section"] = yf_panel.run_cross_section(limit=xs_limit, force=force)
        out["t2.external"] = external.run(force=force)
    if "t3" in tiers:
        out["t3.binance"] = binance.run(interval=intraday_interval, start=intraday_start,
                                        force=force)
    if "t4" in tiers:
        out["t4.scrape"] = t4_scrape.run(force=force, download=download_t4)
    return out


def status() -> dict:
    man = manifest_read()
    fails = json.loads(FAILURES.read_text()) if FAILURES.exists() else {}
    t4 = None
    rep = (FAILURES.parent / "t4_report.json")
    if rep.exists():
        t4 = json.loads(rep.read_text()).get("summary")
    return {"n_datasets": len(man),
            "datasets": {k: {kk: v.get(kk) for kk in ("kind", "source", "n_obs", "freq",
                                                      "date_range")}
                         for k, v in man.items()},
            "failures": fails, "t4_summary": t4}


def main() -> None:
    ap = argparse.ArgumentParser(description="benchmark dataset prefetch")
    ap.add_argument("--tier", default="t1,t2,t3,t4")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--xs-limit", type=int, default=0)
    ap.add_argument("--intraday-interval", default="1h")
    ap.add_argument("--intraday-start", default="2019-01")
    ap.add_argument("--no-t4-download", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        s = status()
        print(json.dumps({"n_datasets": s["n_datasets"], "failures": s["failures"],
                          "t4": s["t4_summary"]}, indent=1))
        for name, meta in sorted(s["datasets"].items()):
            print(f"  {name:<34} {meta['n_obs']:>7} {meta['freq']:<9} "
                  f"{meta['source']}")
        return
    res = prefetch(tuple(t.strip() for t in a.tier.split(",")), force=a.force,
                   xs_limit=a.xs_limit, intraday_interval=a.intraday_interval,
                   intraday_start=a.intraday_start, download_t4=not a.no_t4_download)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
