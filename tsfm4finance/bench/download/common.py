"""Dataset acquisition layer: idempotent, resumable, checksummed.

Every downloader follows the same contract:

* ``data/raw/``            raw vendor files, never deleted, resume-friendly
* ``data/open/<name>/``    the arena format: train/validation/test.parquet + meta.json
* ``data/manifest.json``   what exists, where it came from, its sha256, its shape

A downloader that fails leaves the raw partial file in place and records the
failure in ``data/download_failures.json`` -- the sweep then reports the gap
instead of silently shrinking the grid.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
RAW = DATA / "raw"
OPEN = DATA / "open"
PANELS = OPEN / "_panels"
MANIFEST = DATA / "manifest.json"
FAILURES = DATA / "download_failures.json"

for d in (RAW, OPEN, PANELS):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def retry(times: int = 3, base_delay: float = 2.0, exceptions=(Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            last = None
            for i in range(times):
                try:
                    return fn(*a, **kw)
                except exceptions as e:                      # pragma: no cover
                    last = e
                    if i < times - 1:
                        time.sleep(base_delay * (2**i))
            raise last
        return wrap
    return deco


def http_get(url: str, timeout: int = 90, headers: dict | None = None) -> bytes:
    """GET with curl first, requests as the fallback.

    Several finance data hosts (FRED sits behind Akamai) answer plain curl in
    50 ms while python-requests hangs until the read timeout - a TLS-fingerprint
    filter.  curl is therefore the primary path here; requests remains as a
    fallback for hosts that behave the other way round.  A short backoff avoids
    tripping the throttle twice in a row.
    """
    import subprocess
    ua = (headers or {}).get("User-Agent", "Mozilla/5.0 tsfm4finance-bench/1.0")
    cmd = ["curl", "-sSL", "--http1.1", "--compressed", "-A", ua,
           "--max-time", str(timeout), url]
    for k, v in (headers or {}).items():
        if k.lower() != "user-agent":
            cmd += ["-H", f"{k}: {v}"]
    last = None
    for attempt in range(3):
        try:
            out = subprocess.run(cmd, capture_output=True)
            if out.returncode == 0 and out.stdout:
                return out.stdout
            last = RuntimeError(f"curl rc={out.returncode}: {out.stderr[:200]!r} for {url}")
        except Exception as e:                                     # noqa: BLE001
            last = e
        time.sleep(1.5 * (2**attempt))
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": ua})
        r.raise_for_status()
        return r.content
    except Exception as e:                                         # noqa: BLE001
        raise last or e


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def record_failure(source: str, key: str, error: str) -> None:
    cur = json.loads(FAILURES.read_text()) if FAILURES.exists() else {}
    cur[f"{source}:{key}"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "error": str(error)[:500]}
    FAILURES.write_text(json.dumps(cur, indent=1))


def clear_failure(source: str, key: str) -> None:
    if not FAILURES.exists():
        return
    cur = json.loads(FAILURES.read_text())
    if cur.pop(f"{source}:{key}", None) is not None:
        FAILURES.write_text(json.dumps(cur, indent=1))


def manifest_update(name: str, **fields) -> None:
    cur = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    entry = cur.get(name, {})
    entry.update(fields)
    entry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur[name] = entry
    MANIFEST.write_text(json.dumps(cur, indent=1))


def manifest_read() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


# --------------------------------------------------------------------------- #
SPLIT_DEFAULT = (0.70, 0.15, 0.15)


def write_open_dataset(name: str, df: pd.DataFrame, *, freq: str, domain: str,
                       source: str, license_: str = "open", target: str = "close",
                       split: tuple[float, float, float] = SPLIT_DEFAULT,
                       notes: str = "", task: str = "forecast",
                       extra_meta: dict | None = None) -> Path:
    """Write the arena layout with chronological splits and a truthful meta.json.

    ``df`` must be sorted by ``timestamp`` and may carry the OHLCV columns
    (open/high/low/close/volume/amount); forecasting targets stay on ``target``.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError(f"{name}: needs a timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=[target]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    d = OPEN / name
    d.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].to_parquet(d / "train.parquet")
    df.iloc[n_train:n_train + n_val].to_parquet(d / "validation.parquet")
    df.iloc[n_train + n_val:].to_parquet(d / "test.parquet")
    meta = {
        "name": name, "domain": domain, "task": task, "frequency": freq,
        "n_features": int(df.shape[1] - 1), "feature_names": [c for c in df.columns if c != "timestamp"],
        "n_obs": n,
        "date_range": [str(df.timestamp.min())[:10], str(df.timestamp.max())[:10]],
        "split": {"train": n_train, "validation": n_val, "test": n - n_train - n_val},
        "license": license_, "source": source, "open": True, "requires_download": False,
        "target": target, "lookback": 512, "horizon": 63, "rolling": True,
        "notes": notes,
    }
    if extra_meta:
        meta.update(extra_meta)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    manifest_update(name, kind="dataset", source=source, freq=freq, domain=domain,
                    n_obs=n, columns=meta["feature_names"],
                    date_range=meta["date_range"],
                    sha256=sha256_file(d / "test.parquet"))
    return d


def panel_to_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def dataset_exists(name: str) -> bool:
    d = OPEN / name
    return (d / "train.parquet").exists() and (d / "meta.json").exists()


def dataset_names(domain: str | None = None) -> list[str]:
    """Datasets actually on disk (the sweep plans only against these)."""
    if not OPEN.exists():
        return []
    out = []
    for d in sorted(OPEN.iterdir()):
        if not (d / "train.parquet").exists() or not (d / "meta.json").exists():
            continue
        if domain:
            try:
                meta = json.loads((d / "meta.json").read_text())
            except Exception:                                      # noqa: BLE001
                continue
            if str(meta.get("domain", "")).lower() != domain.lower():
                continue
        out.append(d.name)
    return out


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    d = OPEN / name
    return (pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "validation.parquet"),
            pd.read_parquet(d / "test.parquet"), json.loads((d / "meta.json").read_text()))
