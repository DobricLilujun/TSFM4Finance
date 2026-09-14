"""tsfm_eval — unified evaluation library for financial time-series foundation models."""
from __future__ import annotations

from tsfm_eval.schemas import (
    DatasetMeta,
    Domain,
    EvaluationMode,
    Frequency,
    PredictOutput,
    TaskType,
)
from tsfm_eval.data.loader import load_dataset, load_meta
from tsfm_eval.data.dataset import TSDataset
from tsfm_eval.data.registry import list_datasets, register_dataset
from tsfm_eval.evaluation.evaluator import evaluate
from tsfm_eval.models.base import BaseModel, PredictionResult

__all__ = [
    "BaseModel",
    "DatasetMeta",
    "Domain",
    "EvaluationMode",
    "Frequency",
    "list_datasets",
    "load_dataset",
    "load_meta",
    "PredictOutput",
    "PredictionResult",
    "register_dataset",
    "TaskType",
    "TSDataset",
    "evaluate",
]
