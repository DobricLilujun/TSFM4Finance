"""Core data contracts for Finance TSFMs Arena (Pydantic v2)."""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel as _BM, Field


class TaskType(str, Enum):
    FORECAST = "forecast"
    CLASSIFY = "classify"
    ANOMALY = "anomaly"


class Frequency(str, Enum):
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class Domain(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    INDEX = "index"
    BOND = "bond"
    COMMODITY = "commodity"
    FRAUD = "fraud"
    EVENT = "event"
    GENERAL = "general"


class EvaluationMode(str, Enum):
    OPEN = "open"       # public dataset, results public on leaderboard
    PRIVATE = "private" # closed dataset, results only visible to uploader


class DatasetMeta(_BM):
    name: str
    domain: Domain
    task: TaskType
    frequency: Frequency
    n_features: int
    feature_names: list[str]
    n_obs: int = 0
    date_range: list[str] = Field(default_factory=lambda: ["", ""])
    split: dict[str, list[str]] = Field(default_factory=dict)
    license: str = "unknown"
    source: str = ""
    open: bool = True
    requires_download: bool = False
    # task-specific
    target: str = "close"
    lookback: int = 120
    horizon: int = 12
    rolling: bool = True
    notes: str = ""


class PredictOutput(_BM):
    """Output of a model's predict()."""
    point: Optional[list[float]] = None              # (horizon,) or (horizon, n_targets)
    quantiles: Optional[list[list[float]]] = None    # (K, horizon)
    probabilities: Optional[list[float]] = None      # classify: (n_classes,)
    confidence: Optional[float] = None

    def as_point(self) -> list[float]:
        if self.point is None:
            raise ValueError("PredictOutput has no point prediction")
        return self.point