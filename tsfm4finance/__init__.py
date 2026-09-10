"""Finance TSFMs Arena — a unified benchmark / leaderboard for financial
time-series foundation models (TSFMs).

Package layout:
  tsfm4finance.core    data contracts, metrics, dataset builder, model adapters
  tsfm4finance.backend FastAPI service + encrypted config + model registry
  tsfm4finance.web     static frontend (GitHub Pages)
  tsfm4finance.paths   central path resolution

Run the backend:
  uv run uvicorn tsfm4finance.backend.app:app --port 8000
  # or:  uv run python -m tsfm4finance.backend.app
"""
from tsfm4finance.core import (  # noqa: F401
    DatasetMeta,
    PredictOutput,
    TaskType,
    Frequency,
    Domain,
    EvaluationMode,
    evaluate,
    BaseModel,
    PredictionResult,
)

__version__ = "1.1.0"
