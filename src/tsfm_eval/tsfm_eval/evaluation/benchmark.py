"""Batch benchmark runner over multiple datasets / windows."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from tsfm_eval.data.dataset import TSDataset
from tsfm_eval.evaluation.evaluator import evaluate
from tsfm_eval.models.base import BaseModel
from tsfm_eval.schemas import DatasetMeta, EvaluationMode, LeaderboardEntry, TaskType


def run_benchmark(
    model: BaseModel,
    datasets: Iterable[str | DatasetMeta | TSDataset],
    split: str = "test",
    step: int = 1,
    mode: EvaluationMode = EvaluationMode.OPEN,
) -> list[LeaderboardEntry]:
    """Evaluate a model across multiple datasets using rolling windows.

    Parameters
    ----------
    model: a ``BaseModel`` adapter.
    datasets: iterable of dataset names, ``DatasetMeta``, or ``TSDataset`` objects.
    split: which split to roll over (default ``test``).
    step: rolling-window step.
    mode: ``open`` or ``private``.

    Returns
    -------
    List of ``LeaderboardEntry`` objects, one per dataset.
    """
    from tsfm_eval import load_dataset

    results: list[LeaderboardEntry] = []
    for item in datasets:
        ds = load_dataset(item) if isinstance(item, str) else (
            TSDataset(meta=item) if isinstance(item, DatasetMeta) else item
        )
        assert isinstance(ds, TSDataset)
        if ds.task not in model.supports:
            continue

        all_metrics: list[dict[str, float]] = []
        for context, truth in ds.iter_rolling(split=split, step=step):
            pred = model.predict(context, ds.meta, horizon=ds.horizon)
            metrics = evaluate(pred.to_output(), truth, ds.meta, train_df=ds.train_df)
            all_metrics.append(metrics)

        if not all_metrics:
            # Fallback: predict once on the whole split if rolling windows cannot be formed.
            df = ds.split(split)
            n = len(df)
            if n >= ds.lookback + ds.horizon:
                context = df.iloc[: ds.lookback]
                truth = df.iloc[ds.lookback : ds.lookback + ds.horizon]
            else:
                context = df.iloc[: ds.lookback] if n >= ds.lookback else df
                truth = df.iloc[ds.lookback :]
            pred = model.predict(context, ds.meta, horizon=ds.horizon)
            all_metrics.append(evaluate(pred.to_output(), truth, ds.meta, train_df=ds.train_df))

        avg_metrics = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }

        results.append(
            LeaderboardEntry(
                model_name=model.name,
                dataset=ds.name,
                task=ds.task,
                frequency=ds.frequency,
                domain=ds.meta.domain,
                metrics=avg_metrics,
                score=avg_metrics.get("score", 0.0),
                mode=mode,
                timestamp=datetime.now(timezone.utc).isoformat(),
                is_benchmark=True,
            )
        )

    # Assign ranks by score within this batch.
    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results
