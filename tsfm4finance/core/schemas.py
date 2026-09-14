"""Re-export data contracts from tsfm_eval for backward compatibility."""
from __future__ import annotations

from tsfm_eval.schemas import (
    DatasetMeta,
    Domain,
    EvaluationMode,
    Frequency,
    PredictOutput,
    TaskType,
)

__all__ = [
    "DatasetMeta",
    "Domain",
    "EvaluationMode",
    "Frequency",
    "PredictOutput",
    "TaskType",
]
