"""Datasets: a time series with an honest time split, plus the window sampler the
trainers and the evaluator share.

Three input shapes are accepted, because users have all three:

* a **directory** in the arena layout (``train.parquet`` / ``validation.parquet`` /
  ``test.parquet`` + ``meta.json``);
* a **CSV/Parquet file** with a timestamp column and a target column;
* a **numpy array / pandas Series / DataFrame**.

The split is *always* by time (never a random shuffle), and window sampling is
restricted to a split, so a fine-tune can be proven to have seen only train rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataError

FREQS = ("tick", "second", "minute", "hour", "day", "week", "month", "quarter", "year")


@dataclass
class SeriesDataset:
    """One univariate target plus optional panel columns and a time split."""

    name: str
    target: str = "close"
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    index: pd.DatetimeIndex | None = None
    panel: pd.DataFrame | None = None          # OHLCV etc, same length as values
    splits: dict = field(default_factory=dict)  # {"train": (start, stop), ...}
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(source, target: str | None = None, name: str | None = None,
             freq: str = "day") -> "SeriesDataset":
        if isinstance(source, SeriesDataset):
            return source
        if isinstance(source, (pd.Series, np.ndarray, list)) and not isinstance(source, str):
            arr = np.asarray(source, dtype=float).reshape(-1)
            ds = SeriesDataset(name=name or "array", target=target or "value",
                               values=arr, splits=_even_split(arr))
            ds.meta = {"n_obs": int(len(arr)), "source": "in-memory", "frequency": freq}
            return ds
        p = Path(str(source)).expanduser()
        if p.is_dir():
            return SeriesDataset._from_dir(p, target, name)
        if p.is_file():
            return SeriesDataset._from_file(p, target, name)
        if isinstance(source, pd.DataFrame):
            return SeriesDataset._from_frame(source, target or source.columns[-1], name or "frame")
        if isinstance(source, (pd.Series, np.ndarray, list)):
            arr = np.asarray(source, dtype=float).reshape(-1)
            ds = SeriesDataset(name=name or "array", target=target or "value",
                               values=arr, splits=_even_split(arr))
            ds.meta = {"n_obs": int(len(arr)), "source": "in-memory", "frequency": freq}
            return ds
        raise DataError(f"cannot read a series from {type(source).__name__}")

    @staticmethod
    def _from_dir(d: Path, target: str | None, name: str | None) -> "SeriesDataset":
        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
        target = target or meta.get("target") or "close"
        parts = {}
        for split in ("train", "validation", "test"):
            f = d / f"{split}.parquet"
            if f.exists():
                try:
                    parts[split] = pd.read_parquet(f)
                except ImportError as e:
                    # pandas names 'fastparquet' and buries the actual fix: the parquet
                    # engine is an optional dependency of the *environment*, not of the
                    # model, so the row would otherwise read as a model failure
                    raise DataError(
                        f"{d}/{split}.parquet needs a parquet engine: install pyarrow "
                        f"into this environment (pip install pyarrow). pandas said: "
                        f"{str(e)[:120]}") from e
        if not parts:
            raise DataError(f"{d}: no train/validation/test parquet files")
        full = pd.concat([parts[k] for k in ("train", "validation", "test") if k in parts])
        idx = _index_of(full)
        panel = full.drop(columns=[c for c in (target,) if c in full.columns], errors="ignore")
        splits, acc = {}, 0
        for split, df in parts.items():
            splits[split] = (acc, acc + len(df))
            acc += len(df)
        return SeriesDataset(name=name or meta.get("name") or d.name, target=target,
                             values=np.asarray(full[target], float), index=idx, panel=panel,
                             splits=splits,
                             meta={**meta, "n_obs": int(len(full)), "dir": str(d),
                                   "panel_cols": list(panel.columns)})

    @staticmethod
    def _from_file(f: Path, target: str | None, name: str | None) -> "SeriesDataset":
        df = pd.read_parquet(f) if f.suffix in (".parquet", ".pq") else pd.read_csv(f)
        cand = [c for c in df.columns if c.lower() in
                ("close", "target", "value", "y", "price")]
        target = target or (cand[0] if cand else df.columns[-1])
        return SeriesDataset._from_frame(df, target, name or f.stem)

    @staticmethod
    def _from_frame(df: pd.DataFrame, target: str, name: str) -> "SeriesDataset":
        if target not in df.columns:
            raise DataError(f"{name}: target column {target!r} not in {list(df.columns)}")
        idx = _index_of(df)
        n = len(df)
        tr = int(0.7 * n)
        va = int(0.15 * n)
        splits = {"train": (0, tr), "validation": (tr, tr + va), "test": (tr + va, n)}
        return SeriesDataset(name=name, target=target, values=np.asarray(df[target], float),
                             index=idx, panel=df.drop(columns=[target], errors="ignore"),
                             splits=splits,
                             meta={"n_obs": n, "source": "dataframe",
                                   "panel_cols": [c for c in df.columns if c != target]})

    # ------------------------------------------------------------------ access
    def split(self, which: str = "train") -> np.ndarray:
        if which in ("all", "full"):
            return self.values
        if which not in self.splits:
            raise DataError(f"{self.name}: no {which!r} split (have {list(self.splits)})")
        a, b = self.splits[which]
        return self.values[a:b]

    def panel_window(self, start: int, stop: int) -> pd.DataFrame | None:
        if self.panel is None:
            return None
        df = self.panel.iloc[start:stop].reset_index(drop=True)
        if self.target not in df.columns:
            # the target is normally excluded from the panel (it IS the series), but
            # panel-consuming models (Kronos) need the full OHLCV including close
            df.insert(0, self.target, np.asarray(self.values[start:stop], float))
        if self.index is not None and "timestamp" not in df.columns:
            df.insert(0, "timestamp", self.index[start:stop])
        return df

    def __len__(self) -> int:
        return len(self.values)

    def describe(self) -> dict:
        return {"name": self.name, "target": self.target, "n_obs": len(self.values),
                "splits": {k: (int(v[1] - v[0])) for k, v in self.splits.items()},
                "panel_cols": self.meta.get("panel_cols", []),
                "date_range": [str(self.index[0]), str(self.index[-1])] if self.index is not None else None,
                **{k: v for k, v in self.meta.items() if k in
                   ("frequency", "domain", "source", "license", "task")}}


def as_1d(x) -> np.ndarray:
    """Coerce anything series-shaped to a float 1-D array.

    A ``pandas.Series``, a single-column ``DataFrame``, an ``Index`` or a plain list all
    show up in user code; ``np.asarray`` on a DataFrame iterates *column names*, which is
    how a caller ends up with "could not convert string to float: 'close'".
    """
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    elif hasattr(x, "values") and not isinstance(x, np.ndarray):
        x = np.asarray(x.values)
    a = np.asarray(x, dtype=float).reshape(-1)
    return a


def _index_of(df: pd.DataFrame):
    for c in ("timestamp", "date", "datetime", "time", "ds"):
        if c in df.columns:
            try:
                return pd.DatetimeIndex(pd.to_datetime(df[c]))
            except Exception:                                     # noqa: BLE001
                pass
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    return None


def _even_split(arr, tr=0.7, va=0.15):
    n = len(arr)
    a, b = int(tr * n), int((tr + va) * n)
    return {"train": (0, a), "validation": (a, b), "test": (b, n)}


# --------------------------------------------------------------------- windows
def windows(values: np.ndarray, horizon: int, ctx: int, *, stride: int | None = None,
            max_windows: int | None = None, seed: int = 0, drop_nonfinite: bool = True,
            panel: pd.DataFrame | None = None, with_offsets: bool = False):
    """Sliding ``(context, truth)`` pairs. With ``max_windows`` the subset is chosen
    uniformly at random but returned in time order (reproducibility: same seed,
    same windows).

    ``with_offsets`` appends the window's start position in ``values`` as a third
    element. Callers that need anything *aligned* to the context (a covariate panel, a
    timestamp) must use this: after striding and random subsampling, the position of a
    window in the returned list says nothing about where it came from.
    """
    y = np.asarray(values, dtype=float)
    L = ctx + horizon
    if len(y) < L:
        return []
    out = []
    for i in range(0, len(y) - L + 1, max(1, stride or horizon)):
        c, t = y[i:i + ctx], y[i + ctx:i + L]
        if drop_nonfinite and not (np.all(np.isfinite(c)) and np.all(np.isfinite(t))):
            continue
        item = [c.astype(np.float32), t.astype(np.float32)]
        if panel is not None:
            item.append(panel.iloc[i:i + ctx])
        if with_offsets:
            item.append(i)
        out.append(tuple(item))
    if max_windows and len(out) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(out), size=max_windows, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


def train_windows(dataset: SeriesDataset, horizon: int, ctx: int, *,
                  split: str = "train", stride: int | None = None,
                  max_windows: int | None = None, seed: int = 0,
                  with_panel: bool = False):
    """Windows taken **strictly inside** ``split``.

    Returns ``(windows, info)``; ``info['split_end']`` is what a checkpoint records
    so an adapted model can later be proven never to have seen the evaluation span.
    """
    a, b = (0, len(dataset.values)) if split in ("all", "full") else dataset.splits[split]
    y = dataset.values[a:b]
    panel = dataset.panel.iloc[a:b] if (with_panel and dataset.panel is not None) else None
    w = windows(y, horizon, ctx, stride=stride, max_windows=max_windows, seed=seed, panel=panel)
    info = {"series": dataset.name, "target": dataset.target, "split": split,
            "n_obs": int(b - a), "split_start_index": int(a), "split_end_index": int(b),
            "n_windows": len(w), "horizon": int(horizon), "ctx": int(ctx),
            "with_panel": bool(with_panel)}
    if dataset.index is not None and len(dataset.index) >= b:
        info["split_end"] = str(dataset.index[b - 1])
    if not w:
        raise DataError(f"{dataset.name}: no train windows "
                        f"(need >= {ctx + horizon} obs in {split}, have {b - a})")
    return w, info
