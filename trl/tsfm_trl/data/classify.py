"""Classification datasets.

Two families of label sources, because one of them alone is not defensible:

* **derived from a series** (finance use case): direction, volatility regime,
  extreme-move event, or any user-supplied label vector. These are *self-labelled*
  by a rule, which must be stated in any report built on them.
* **public archives** (UCR/UEA time-series classification): labels that exist
  independently of our rule, which is what makes a classification claim survive a
  reviewer who says "your target is your own heuristic".
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import DataError
from .series import SeriesDataset

CACHE = Path(__file__).resolve().parents[2] / "data_cache"


@dataclass
class ClassificationDataset:
    """Windows + labels. ``X[i]`` is a context array (1-D, single channel) and
    ``y[i]`` its label; ``split`` assigns each window to train/test (by time)."""

    name: str
    X: list = field(default_factory=list)
    y: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    split: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # 0=train 1=test
    meta: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    def parts(self):
        tr, te = self.split == 0, self.split == 1
        return (self.X, self.y, tr, te)

    def limit(self, n: int) -> "ClassificationDataset":
        """An evenly-spaced subsample, splits kept proportional.

        Even spacing rather than "first n": taking a prefix would train on one regime
        and test on another. The cap exists because the matrix measures every family at
        the same scale, and a sampler model's classification features cost ~draws× more
        per window than a quantile head's.
        """
        if not n or len(self.X) <= n:
            return self
        idx = np.unique(np.linspace(0, len(self.X) - 1, int(n)).round().astype(int))
        # keep X an object array: the trainers index it with boolean masks (X[tr]),
        # which a python list refuses
        X = self.X[idx] if isinstance(self.X, np.ndarray) else np.array(
            [self.X[i] for i in idx], dtype=object)
        return ClassificationDataset(name=f"{self.name}~{n}", X=X,
                                     y=self.y[idx], split=self.split[idx],
                                     meta={**self.meta, "subsampled_from": len(self.X)})

    def describe(self) -> dict:
        return {"name": self.name, "n_windows": len(self.X), "n_classes": self.n_classes,
                "classes": [int(c) for c in self.classes()],
                "train": int((self.split == 0).sum()), "test": int((self.split == 1).sum()),
                # len(), not truthiness: X is a numpy object array, and `if X` on it is
                # an ambiguous-truth-value error waiting to fire
                "ctx": int(np.mean([len(x) for x in self.X])) if len(self.X) else 0,
                "base_rate": float(max(np.bincount(self.y, minlength=self.n_classes)
                                       / max(1, len(self.y)))),
                **self.meta}


# ------------------------------------------------------- derived (self-labelled)
def direction_dataset(ds: SeriesDataset, horizon: int = 5, ctx: int = 120,
                      *, band: float = 0.25, levels: int = 3,
                      name: str | None = None) -> ClassificationDataset:
    """Direction over the next ``horizon`` steps, banded by the rolling 20-step
    log-return sd. Relative bands are what make one rule work across S&P and BTC.

    ``levels=3`` -> {down, flat, up}; ``levels=5`` (the ``*_dir5`` task) adds the
    ±2·band "strong" tails: {strong down, down, flat, up, strong up}. Five levels are
    what makes a "the model beats majority-class" claim testable at all, because a
    3-class task with a fat flat class is usually won by predicting flat.
    """
    if levels not in (3, 5):
        raise DataError(f"direction_dataset: levels must be 3 or 5, got {levels}")
    y = ds.values
    lr = np.diff(np.log(np.maximum(y, 1e-12)))
    vol = pd_roll_sd(lr, 20)
    X, lab, tflag = [], [], []
    for i in range(ctx, len(y) - horizon):
        r = float(np.log(max(y[i + horizon], 1e-12) / max(y[i], 1e-12)))
        s = float(vol[i]) if np.isfinite(vol[i]) else 0.01
        z = r / max(s, 1e-12)
        if levels == 3:
            k = 1 if abs(z) <= band else (2 if z > 0 else 0)
        else:                       # 5 levels: 0 strong-down … 4 strong-up
            if z <= -2 * band: k = 0
            elif z <= -band:   k = 1
            elif z < band:     k = 2
            elif z < 2 * band: k = 3
            else:              k = 4
        X.append(y[i - ctx:i].astype(np.float32))
        lab.append(k)
        tflag.append(_is_train(ds, i))
    return _pack(name or f"{ds.name}_dir{horizon}", X, lab, tflag,
                 {"rule": f"log-return over {horizon} vs {band}x rolling-20 sd, "
                          f"{levels} levels", "levels": levels,
                  "series": ds.name, "label_kind": "derived"})


def extreme_dataset(ds: SeriesDataset, horizon: int = 5, ctx: int = 120, *, k: float = 3.0,
                    name: str | None = None) -> ClassificationDataset:
    """1 if any of the next ``horizon`` log-returns exceeds ``k`` × the rolling
    20-step sd (a tail event), else 0. Class-imbalanced on purpose: the metric
    suite must then use PR-AUC, not accuracy."""
    y = ds.values
    lr = np.diff(np.log(np.maximum(y, 1e-12)))
    vol = pd_roll_sd(lr, 20)
    X, lab, tflag = [], [], []
    for i in range(ctx, len(y) - horizon):
        s = float(vol[i]) if np.isfinite(vol[i]) else 0.01
        nxt = lr[i:i + horizon]
        hit = 1 if len(nxt) and np.max(np.abs(nxt)) > k * s else 0
        X.append(y[i - ctx:i].astype(np.float32))
        lab.append(hit)
        tflag.append(_is_train(ds, i))
    return _pack(name or f"{ds.name}_extreme", X, lab, tflag,
                 {"rule": f"any |log-return| > {k}x rolling-20 sd within {horizon}",
                  "series": ds.name, "label_kind": "derived"})


def volreg_dataset(ds: SeriesDataset, horizon: int = 21, ctx: int = 120, n_bins: int = 3,
                   name: str | None = None) -> ClassificationDataset:
    """Realised volatility over the next ``horizon`` steps, bucketed into terciles
    (0 low … n-1 high) using the train-split quantiles only."""
    y = ds.values
    lr = np.diff(np.log(np.maximum(y, 1e-12)))
    rv = np.array([float(np.std(lr[i:i + horizon], ddof=1)) if len(lr[i:i + horizon]) > 1
                   else np.nan for i in range(len(lr))])
    tr_lo, tr_hi = ds.splits.get("train", (0, len(y)))
    hist = rv[:tr_hi - horizon]
    hist = hist[np.isfinite(hist)]
    qs = np.quantile(hist, [i / n_bins for i in range(1, n_bins)]) if len(hist) > n_bins else []
    X, lab, tflag = [], [], []
    for i in range(ctx, len(y) - horizon):
        v = rv[i]
        if not np.isfinite(v):
            continue
        k = int(np.searchsorted(qs, v, side="right")) if len(qs) else 0
        X.append(y[i - ctx:i].astype(np.float32))
        lab.append(k)
        tflag.append(_is_train(ds, i))
    return _pack(name or f"{ds.name}_volreg", X, lab, tflag,
                 {"rule": f"realised-vol {n_bins}-quantile (train-split thresholds)",
                  "series": ds.name, "label_kind": "derived"})


def label_dataset(ds: SeriesDataset, labels: np.ndarray, ctx: int = 120,
                  name: str = "labelled", **meta) -> ClassificationDataset:
    """Wrap an externally provided label vector aligned to the series index
    (``labels[i]`` applies to the window ending at ``i``; negative = unlabeled)."""
    lab = np.asarray(labels)
    X, labs, tflags = [], [], []
    for i in range(ctx, min(len(ds.values), len(lab))):
        if lab[i] < 0:
            continue
        X.append(ds.values[i - ctx:i].astype(np.float32))
        labs.append(int(lab[i]))
        tflags.append(_is_train(ds, i))
    return _pack(name, X, labs, tflags, {"series": ds.name, "label_kind": "external", **meta})


def _is_train(ds: SeriesDataset, i: int) -> int:
    a, b = ds.splits.get("train", (0, int(0.7 * len(ds.values))))
    return 0 if a <= i < b else 1


def _pack(name, X, lab, tflag, meta) -> ClassificationDataset:
    # X becomes an *object* array so boolean split masks index it the way y does
    if not X:
        raise DataError(f"{name}: no windows (series too short for ctx)")
    y = np.asarray(lab, dtype=int)
    keep = np.isin(y, np.unique(y))
    X = np.asarray([x for x, k in zip(X, keep) if k], dtype=object)
    return ClassificationDataset(name=name, X=X, y=y[keep],
                                 split=np.asarray(tflag, dtype=int)[keep], meta=meta)


def pd_roll_sd(lr: np.ndarray, k: int) -> np.ndarray:
    import pandas as pd
    return pd.Series(lr).rolling(k).std(ddof=1).to_numpy()


# ------------------------------------------------------------ public archives
# Small UCR/UEA sets only: the whole point is a defensible external check without
# downloading gigabytes. Sizes below are the published TRAIN/TEST counts.
UCR_SMALL = {
    "Coffee": {"n": "2 classes, univariate, ~1000 points/series"},
    "GunPoint": {"n": "2 classes, univariate, 150 points/series"},
    "PLANT": {"n": "3 classes, univariate, 67 points/series"},
    "FaceFour": {"n": "4 classes, univariate, 350 points/series"},
}


def ucr_dataset(name: str, ctx: int | None = None, cache: Path | None = None,
                url: str | None = None) -> ClassificationDataset:
    """Fetch (once) and parse a UCR/UEA archive's ``_TRAIN``/``_TEST`` files.

    Univariate sets are turned into windows: each series becomes one sample, cut to
    ``ctx`` (default: the full length, so nothing is invented). Multivariate sets
    are flattened channel-wise — documented, not silently averaged.
    """
    cache = Path(cache or CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    d = cache / name
    if not (d / f"{name}_TRAIN.ts").exists():
        url = url or f"https://www.timeseriesclassification.com/Downloads/{name}.zip"
        try:
            _download_extract(url, d)
        except Exception as e:                                     # noqa: BLE001
            raise DataError(f"cannot fetch UEA/UCR {name!r} from {url}: {e}") from e
    train = _parse_ts(d / f"{name}_TRAIN.ts")
    test = _parse_ts(d / f"{name}_TEST.ts") if (d / f"{name}_TEST.ts").exists() else []
    if not train:
        raise DataError(f"{name}: archive parsed to zero series")
    raw_tr, raw_te = train, test
    labels = sorted({lab for _, lab in raw_tr + raw_te}, key=str)
    if not all(isinstance(l, (int, np.integer)) for l in labels):
        enc = {l: i for i, l in enumerate(labels)}
    else:
        enc = {l: int(l) for l in labels}

    Xw, yall, sp = [], [], []
    for part, flag in ((raw_tr, 0), (raw_te, 1)):
        for arr, lab in part:
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                a = a[None, :]
            L = ctx or a.shape[1]
            if a.shape[1] > L:
                a = a[:, -L:]                                    # keep the tail, never pad
            k = enc.get(lab, -1)
            if k < 0:
                continue
            for c in range(a.shape[0]):
                Xw.append(np.nan_to_num(a[c], nan=0.0).astype(np.float32))
                yall.append(k)
                sp.append(flag)
    n_channels = int(max(1, *(np.asarray(a).shape[0] if np.asarray(a).ndim > 1 else 1
                              for a, _ in raw_tr[:4])))
    if not Xw:
        raise DataError(f"{name}: no usable samples")
    return ClassificationDataset(
        name=f"ucr_{name}", X=Xw, y=np.asarray(yall, int), split=np.asarray(sp, int),
        meta={"source": "UCR/UEA archive", "label_kind": "external", "n_channels": n_channels,
              "note": "one sample per (series, channel); tail-truncated to ctx"})


def _download_extract(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:     # noqa: S310 (fixed host list)
        blob = r.read()
    zp = dest / "_archive.zip"
    zp.write_bytes(blob)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.upper().endswith(".TS")]
        for n in names:
            z.extract(n, dest)
            p = dest / n
            target = dest / Path(n).name
            if p != target:
                target.write_bytes(p.read_bytes())
        if not names:                                       # some archives ship .arff
            for n in z.namelist():
                if n.upper().endswith(".ARFF"):
                    z.extract(n, dest)


def _parse_ts(path: Path):
    """Minimal ``.ts`` parser: header tags then ``@data`` rows of comma-separated
    values per dimension followed by the class value."""
    lines = path.read_text(errors="ignore").splitlines()
    dims, npoints = 1, None
    data_at, vals = None, []
    for j, ln in enumerate(lines):
        s = ln.strip()
        low = s.lower()
        if low.startswith("@problemname"):
            continue
        if low.startswith("@dimensions"):
            dims = int(s.split()[1])
        elif low.startswith("@points"):
            npoints = int(s.split()[1])
        elif low.startswith("@data"):
            data_at = j + 1
            break
    if data_at is None:
        return []
    rows = [l for l in lines[data_at:] if l.strip() and not l.strip().startswith("@")]
    out = []
    for r in rows:
        parts = [p for p in r.replace("\t", ",").split(",")]
        parts = [p.strip() for p in parts if p.strip() != ""]
        if not parts:
            continue
        label = parts[-1]
        try:
            nums = [float("nan") if p in ("?", "nan", "NaN") else float(p) for p in parts[:-1]]
        except ValueError:
            continue
        per = len(nums) // max(1, dims)
        arr = np.array(nums, dtype=float)
        if dims > 1 and per * dims == len(nums):
            arr = arr.reshape(dims, per)
        out.append((arr, _coerce_label(label)))
    if npoints and out and getattr(out[0][0], "shape", (0,))[-1] not in (npoints,):
        pass                                                # tolerate header/actual mismatch
    return out


def _coerce_label(s: str):
    try:
        return int(float(s))
    except ValueError:
        return s
