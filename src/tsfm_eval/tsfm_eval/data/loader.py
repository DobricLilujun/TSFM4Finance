"""Low-level dataset I/O: load meta.json and parquet splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsfm_eval.config import get_data_root
from tsfm_eval.schemas import DatasetMeta


def _dataset_dir(name: str, data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    for bucket in ("open", "closed"):
        d = root / bucket / name
        if (d / "meta.json").exists():
            return d
    raise FileNotFoundError(f"Dataset '{name}' not found under {root}/open or {root}/closed")


def load_meta(name: str, data_root: Path | None = None) -> DatasetMeta:
    """Load the ``meta.json`` for a dataset."""
    p = _dataset_dir(name, data_root) / "meta.json"
    return DatasetMeta.model_validate_json(p.read_text(encoding="utf-8"))


def load_split(name: str, split: str, data_root: Path | None = None) -> pd.DataFrame:
    """Load one split (train / validation / test) as a DataFrame."""
    d = _dataset_dir(name, data_root)
    candidates = {
        split: d / f"{split}.parquet",
        "validation": d / "validation.parquet",
        "val": d / "validation.parquet",
    }
    path = candidates.get(split)
    if path is None or not path.exists():
        # fallback: try the literal name
        path = d / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split '{split}' not found for dataset '{name}' at {d}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_dataset(name: str, data_root: Path | None = None) -> "tsfm_eval.data.dataset.TSDataset":  # type: ignore[name-defined]
    """Load a dataset with all splits as a :class:`TSDataset`."""
    from tsfm_eval.data.dataset import TSDataset

    return TSDataset(name=name, data_root=data_root)
