"""Composite leaderboard score (0..1, higher is better)."""
from __future__ import annotations

from tsfm_eval.schemas import TaskType


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_forecast(m: dict[str, float]) -> float:
    s = 0.35 * (1 - _clip01(m.get("mase", 1.0)))
    s += 0.25 * m.get("directional_accuracy", 0.5)
    s += 0.20 * max(0.0, m.get("return_rank_corr", 0.0))
    s += 0.20 * (1 - _clip01(m.get("smape", 100.0) / 100.0))
    return round(_clip01(s), 4)


def _score_classify(m: dict[str, float]) -> float:
    return round(_clip01(0.6 * m.get("f1", 0.0) + 0.4 * m.get("auc", 0.5)), 4)


def _score_anomaly(m: dict[str, float]) -> float:
    return round(_clip01(0.6 * m.get("f1", 0.0) + 0.4 * m.get("auc", 0.5)), 4)


_COMPOSITE = {
    TaskType.FORECAST: _score_forecast,
    TaskType.CLASSIFY: _score_classify,
    TaskType.ANOMALY: _score_anomaly,
}


def composite_score(task: TaskType, m: dict[str, float]) -> float:
    return _COMPOSITE[task](m)
