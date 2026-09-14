"""Unified Dataset interface for time-series evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from tsfm_eval.config import get_data_root
from tsfm_eval.data.loader import load_meta, load_split
from tsfm_eval.schemas import DatasetMeta


class TSDataset:
    """A single arena dataset with train/validation/test splits.

    This is the standard object passed to models and evaluators.  It can be
    constructed from the on-disk parquet layout or directly from DataFrames
    (useful for in-memory tests or uploaded private data).
    """

    def __init__(
        self,
        name: str | None = None,
        data_root: Path | None = None,
        meta: DatasetMeta | None = None,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ):
        self._data_root = data_root or get_data_root()
        self._name = name

        if meta is None:
            if name is None:
                raise ValueError("Either 'name' or 'meta' must be provided")
            self._meta = load_meta(name, self._data_root)
        else:
            self._meta = meta
            if name is not None:
                self._meta.name = name

        self._train = train_df
        self._val = val_df
        self._test = test_df

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._meta.name

    @property
    def meta(self) -> DatasetMeta:
        return self._meta

    @property
    def task(self):
        return self._meta.task

    @property
    def target(self) -> str:
        return self._meta.target

    @property
    def lookback(self) -> int:
        return self._meta.lookback

    @property
    def horizon(self) -> int:
        return self._meta.horizon

    @property
    def frequency(self):
        return self._meta.frequency

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self._meta.feature_names if c != "timestamp"]

    @property
    def train_df(self) -> pd.DataFrame:
        if self._train is None:
            self._train = load_split(self.name, "train", self._data_root)
        return self._train

    @property
    def val_df(self) -> pd.DataFrame:
        if self._val is None:
            self._val = load_split(self.name, "validation", self._data_root)
        return self._val

    @property
    def test_df(self) -> pd.DataFrame:
        if self._test is None:
            self._test = load_split(self.name, "test", self._data_root)
        return self._test

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def split(self, name: str) -> pd.DataFrame:
        """Return one of 'train', 'validation', 'test'."""
        if name == "train":
            return self.train_df
        if name in ("validation", "val"):
            return self.val_df
        if name == "test":
            return self.test_df
        raise ValueError(f"Unknown split '{name}'")

    def target_values(self, split: str) -> np.ndarray:
        """Return the target column for a split as a float array."""
        df = self.split(split)
        col = self.target
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
        return df.select_dtypes(include="number").iloc[:, -1].to_numpy(dtype=float)

    def feature_values(self, split: str) -> np.ndarray:
        """Return all numeric feature columns (excluding timestamp) for a split."""
        df = self.split(split)
        return df.drop(columns=[c for c in ("timestamp",) if c in df.columns]).to_numpy(dtype=float)

    def iter_rolling(
        self,
        split: str = "test",
        lookback: int | None = None,
        horizon: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (context, truth) windows over a split.

        context has ``lookback`` rows, truth has ``horizon`` rows.  ``step``
        controls how many rows the window advances each iteration.
        """
        lb = lookback if lookback is not None else self.lookback
        hz = horizon if horizon is not None else self.horizon
        df = self.split(split).reset_index(drop=True)
        n = len(df)
        # If the split is too short for the declared lookback, still produce
        # at least one window so evaluation does not silently empty out.
        if n < hz + 1:
            return
        if n < lb + hz:
            ctx = df.iloc[: n - hz]
            truth = df.iloc[n - hz :]
            yield ctx, truth
            return
        for i in range(lb, n - hz + 1, step):
            context = df.iloc[i - lb : i]
            truth = df.iloc[i : i + hz]
            yield context, truth

    def to_numpy(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        """Return (features, target) arrays for a split."""
        return self.feature_values(split), self.target_values(split)
