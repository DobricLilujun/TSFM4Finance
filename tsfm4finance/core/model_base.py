"""Base model interface for Finance TSFMs Arena.

A model adapter wraps any forecasting / classification / anomaly model behind
the unified predict() contract in docs/02_unified_interface_spec.md §3.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .schemas import DatasetMeta, PredictOutput, TaskType


@dataclass
class PredictionResult:
    """Convenience wrapper returned by BaseModel.predict()."""
    point: Optional[list[float]] = None
    quantiles: Optional[list[list[float]]] = None
    probabilities: Optional[list[float]] = None
    confidence: Optional[float] = None

    def to_output(self) -> PredictOutput:
        return PredictOutput(
            point=self.point,
            quantiles=self.quantiles,
            probabilities=self.probabilities,
            confidence=self.confidence,
        )


class BaseModel(abc.ABC):
    """Unified model interface.

    Subclasses implement predict(). fit() is optional (TSFMs are typically
    zero-shot / pre-trained).
    """

    def __init__(self, name: str, supports: set[str] | None = None):
        self.name = name
        self.supports: set[TaskType] = {
            TaskType(t) if isinstance(t, str) else t
            for t in (supports or {"forecast"})
        }

    def fit(self, train_df: pd.DataFrame, meta: DatasetMeta) -> None:  # noqa: D401
        """Optional training / warmup. Default: no-op (zero-shot)."""
        return None

    @abc.abstractmethod
    def predict(
        self,
        context_df: pd.DataFrame,
        meta: DatasetMeta,
        horizon: int | None = None,
    ) -> PredictionResult:
        """Predict over `context_df` (lookback rows). Return PredictionResult."""
        ...

    def predict_and_evaluate(
        self,
        context_df: pd.DataFrame,
        truth_df: pd.DataFrame,
        meta: DatasetMeta,
        train_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Predict then evaluate against ground truth (convenience for benchmark)."""
        from .metrics import evaluate

        res = self.predict(context_df, meta)
        return evaluate(res.to_output(), truth_df, meta, train_df)