"""Dataset registry: discover and register datasets under the data root."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tsfm_eval.config import get_data_root
from tsfm_eval.schemas import DatasetMeta


BUCKETS = ("open", "closed")


def _scan(data_root: Path) -> dict[str, Path]:
    """Return mapping dataset_name -> directory path for every meta.json found."""
    found: dict[str, Path] = {}
    root = Path(data_root)
    for bucket in BUCKETS:
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            continue
        for d in sorted(bucket_dir.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                found[d.name] = d
    return found


def list_datasets(open_only: bool = False, data_root: Path | None = None) -> list[DatasetMeta]:
    """Return metadata for all discovered datasets."""
    root = data_root or get_data_root()
    out = []
    for name, d in _scan(root).items():
        meta = DatasetMeta.model_validate_json((d / "meta.json").read_text(encoding="utf-8"))
        if open_only and not meta.open:
            continue
        out.append(meta)
    return out


def list_dataset_names(open_only: bool = False, data_root: Path | None = None) -> list[str]:
    """Return dataset names only."""
    return [m.name for m in list_datasets(open_only=open_only, data_root=data_root)]


def get_dataset_dir(name: str, data_root: Path | None = None) -> Path:
    """Return the on-disk directory for a dataset."""
    root = data_root or get_data_root()
    found = _scan(root)
    if name not in found:
        raise FileNotFoundError(f"Dataset '{name}' not found under {root}")
    return found[name]


def register_dataset(
    name: str,
    meta: DatasetMeta,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    data_root: Path | None = None,
    closed: bool = False,
) -> Path:
    """Write a dataset to disk in the standard parquet + meta.json layout.

    Parameters
    ----------
    name: dataset name.
    meta: metadata object.
    train/val/test: split DataFrames.
    data_root: optional override for the data root.
    closed: if True, write to ``closed/`` bucket, otherwise ``open/``.
    """
    root = data_root or get_data_root()
    bucket = "closed" if closed else "open"
    d = root / bucket / name
    d.mkdir(parents=True, exist_ok=True)

    meta.name = name
    meta.n_obs = len(train) + len(val) + len(test)
    if "timestamp" in train.columns:
        all_ts = pd.concat([train["timestamp"], val["timestamp"], test["timestamp"]])
        meta.date_range = [str(all_ts.min()), str(all_ts.max())]
        meta.split = {
            "train": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
            "validation": [str(val["timestamp"].iloc[0]), str(val["timestamp"].iloc[-1])],
            "test": [str(test["timestamp"].iloc[0]), str(test["timestamp"].iloc[-1])],
        }

    train.to_parquet(d / "train.parquet")
    val.to_parquet(d / "validation.parquet")
    test.to_parquet(d / "test.parquet")
    (d / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return d


def upload_private_data(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    meta: DatasetMeta | None = None,
    data_root: Path | None = None,
) -> Path:
    """Replace the synthetic stub of a closed dataset with real private data."""
    root = data_root or get_data_root()
    if meta is None:
        existing = get_dataset_dir(name, root) / "meta.json"
        meta = DatasetMeta.model_validate_json(existing.read_text(encoding="utf-8"))
    meta.requires_download = False
    meta.open = False
    return register_dataset(name, meta, train, val, test, data_root=root, closed=True)
