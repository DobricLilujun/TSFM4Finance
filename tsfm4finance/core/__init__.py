"""Finance TSFMs Arena — unified interface & data contracts.

See docs/02_unified_interface_spec.md for the full specification.
"""
from .schemas import (
    DatasetMeta,
    PredictOutput,
    TaskType,
    Frequency,
    Domain,
    EvaluationMode,
)
from .metrics import evaluate
from .model_base import BaseModel, PredictionResult

__all__ = [
    "DatasetMeta",
    "PredictOutput",
    "TaskType",
    "Frequency",
    "Domain",
    "EvaluationMode",
    "evaluate",
    "BaseModel",
    "PredictionResult",
]

__version__ = "1.0.0"