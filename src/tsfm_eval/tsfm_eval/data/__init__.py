"""Data loading, registry and unified Dataset interface."""
from __future__ import annotations

from tsfm_eval.data.dataset import TSDataset
from tsfm_eval.data.loader import load_dataset, load_meta
from tsfm_eval.data.registry import list_datasets, register_dataset

__all__ = ["TSDataset", "load_dataset", "load_meta", "list_datasets", "register_dataset"]
