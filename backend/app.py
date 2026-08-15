"""Finance TSFMs Arena — FastAPI backend.

Endpoints (see backend/README.md):
  GET  /api/health
  GET  /api/datasets
  GET  /api/datasets/{name}
  GET  /api/models
  POST /api/evaluate      {dataset, model, mode, horizon?, lookback?}
  POST /api/submit       {dataset, model_name, mode}   (upload-model mock)
  GET  /api/leaderboard  (?task=&domain=)
  GET  /api/leaderboard/{model}

Run:  uv run python -m backend.app   (or)  uv run uvicorn backend.app:app --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Make the project root importable so `import arena` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as _BM

from arena.schemas import DatasetMeta, TaskType
from arena.models import _build, available_models
from arena.metrics import evaluate as run_eval

DATA = Path(__file__).resolve().parent.parent / "data"


def _to_native(obj):
    """Recursively convert numpy types -> native Python so JSON serialises.

    NaN / Infinity are not valid JSON and Starlette renders with
    allow_nan=False, so they are normalised to 0.0.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        v = obj.item()
    else:
        v = obj
    if isinstance(v, float):
        import math
        if math.isnan(v) or math.isinf(v):
            return 0.0
    return v

app = FastAPI(title="Finance TSFMs Arena", version="0.1.0")
# Allow the static frontend (served from another origin / file://) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EvaluateReq(_BM):
    dataset: str
    model: str
    mode: str = "open"          # 'open' -> public leaderboard; 'private' -> not stored
    horizon: Optional[int] = None
    lookback: Optional[int] = None


class SubmitReq(_BM):
    dataset: str
    model_name: str
    mode: str = "open"          # 'private' for closed/user's-own model
    model_ref: Optional[str] = None   # placeholder for uploaded model artifact


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _load_dataset(name: str) -> DatasetMeta:
    for bucket in ("open", "closed"):
        p = DATA / bucket / name / "meta.json"
        if p.exists():
            return DatasetMeta.model_validate_json(p.read_text())
    raise HTTPException(404, f"dataset '{name}' not found")


def _load_splits(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = DATA / ("open" if _load_dataset(name).open else "closed") / name
    train = pd.read_parquet(d / "train.parquet")
    val = pd.read_parquet(d / "validation.parquet")
    test = pd.read_parquet(d / "test.parquet")
    return train, val, test


def _rolling_windows(train, val, test, lookback: int,
                     horizon: int, step: Optional[int] = None):
    """Produce (context, truth) pairs by rolling over the val+test tail.

    Context is the real `lookback` points immediately preceding each window;
    truth is the `horizon` points the model must predict. Rolling uses the
    val+test concatenation so there is always enough history even when the
    held-out `test` block is shorter than `lookback`.
    """
    cont = pd.concat([val, test], ignore_index=True) if len(val) else test
    n = len(cont)
    step = step or max(1, horizon)
    windows = []
    start = 0
    # require enough history before the window so context is real data
    start = max(0, n - lookback)
    while start + horizon <= n:
        ctx = cont.iloc[max(0, start - lookback):start]
        truth = cont.iloc[start:start + horizon]
        if len(ctx) >= 1:
            windows.append((ctx, truth))
        start += step
    if not windows:  # last resort: single window on the whole series
        ctx = cont.iloc[:max(1, n - horizon)]
        truth = cont.iloc[max(0, n - horizon):]
        if len(truth) >= 1:
            windows.append((ctx, truth))
    return windows


def _evaluate_dataset(meta: DatasetMeta, model_name: str,
                      horizon: Optional[int], lookback: Optional[int]) -> dict:
    """Run a model over rolling test windows; average the metrics."""
    from arena.schemas import PredictOutput
    from arena.model_base import PredictionResult

    train, val, test = _load_splits(meta.name)
    horizon = horizon or meta.horizon
    lookback = lookback or meta.lookback

    model = _build(model_name)

    # For classify/anomaly we evaluate on the tail directly; for forecast we
    # roll. Keep it robust: fall back to a single window if rolling fails.
    windows = _rolling_windows(train, val, test, lookback, horizon)
    if not windows:
        windows = [(train.iloc[-lookback:], test.iloc[:horizon])]

    per_point = []
    for ctx, truth in windows:
        try:
            pred = model.predict(ctx, meta, horizon=horizon)
            if meta.task == TaskType.FORECAST:
                metrics = run_eval(PredictOutput(point=pred.point), truth, meta,
                                   train_df=train)
            else:
                metrics = run_eval(PredictOutput(point=pred.point,
                                                probabilities=pred.probabilities),
                                   truth, meta, train_df=train)
            per_point.append(metrics)
        except Exception:
            continue

    if not per_point:
        raise HTTPException(500, f"evaluation failed for {model_name} on {meta.name}")

    # average across rolling points
    keys = per_point[0].keys()
    avg = {k: float(np.nanmean([m[k] for m in per_point])) for k in keys}
    avg["n_points"] = len(per_point)
    return avg


# --------------------------------------------------------------------------- #
# Leaderboard storage
# --------------------------------------------------------------------------- #
class Leaderboard:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = []
        self.load()

    def load(self):
        if self.path.exists():
            import json
            self.records = json.loads(self.path.read_text())

    def save(self):
        import json
        self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2))

    def add(self, record: dict):
        self.records.append(record)
        self.save()

    def query(self, task: Optional[str] = None,
              domain: Optional[str] = None) -> list[dict]:
        out = self.records
        if task:
            out = [r for r in out if r.get("task") == task]
        if domain:
            out = [r for r in out if r.get("domain") == domain]
        return sorted(out, key=lambda r: r.get("score", 0), reverse=True)


LB = Leaderboard(Path(__file__).resolve().parent / "leaderboard.json")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "datasets": len(_list_datasets()), "leaderboard": len(LB.records)}


def _list_datasets() -> list[DatasetMeta]:
    out = []
    for bucket in ("open", "closed"):
        for d in (DATA / bucket).glob("*"):
            p = d / "meta.json"
            if p.exists():
                out.append(DatasetMeta.model_validate_json(p.read_text()))
    return out


@app.get("/api/datasets")
def datasets():
    return [m.model_dump() for m in _list_datasets()]


@app.get("/api/datasets/{name}")
def dataset_detail(name: str):
    meta = _load_dataset(name)
    try:
        train, val, test = _load_splits(name)
        counts = {"train": len(train), "validation": len(val), "test": len(test)}
    except Exception:
        counts = {}
    return {**meta.model_dump(), "split_rows": counts}


@app.get("/api/models")
def models():
    """Return the available model adapters as a list of {name, info}."""
    out = []
    for name, info in available_models().items():
        entry = {"name": name}
        entry.update(info) if isinstance(info, dict) else entry.update({"info": info})
        out.append(entry)
    return out


@app.post("/api/evaluate")
def evaluate(req: EvaluateReq):
    meta = _load_dataset(req.dataset)
    if req.model not in available_models():
        raise HTTPException(404, f"unknown model '{req.model}' (available: {sorted(available_models())})")
    try:
        metrics = _evaluate_dataset(meta, req.model, req.horizon, req.lookback)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"evaluation error: {e}")

    record = {
        "dataset": meta.name, "model": req.model, "mode": req.mode,
        "domain": meta.domain.value, "task": meta.task.value,
        "frequency": meta.frequency.value,
        "score": metrics.get("score", 0.0),
        "metrics": metrics,
    }
    # private mode -> return but do NOT publish
    if req.mode != "private":
        LB.add(record)
    return _to_native(record)


@app.post("/api/submit")
def submit(req: SubmitReq):
    """Local mock of 'upload your model and let the arena evaluate it'.

    In a real deployment this endpoint would receive a model artifact /
    inference service (req.model_ref) and run it here. Locally we resolve the
    named model from the registry and evaluate it, exactly like /evaluate.
    """
    meta = _load_dataset(req.dataset)
    if req.model_name not in available_models():
        raise HTTPException(404, f"unknown model '{req.model_name}' (available: {sorted(available_models())})")
    try:
        metrics = _evaluate_dataset(meta, req.model_name, None, None)
    except Exception as e:
        raise HTTPException(500, f"evaluation error: {e}")

    record = {
        "dataset": meta.name, "model": req.model_name, "mode": req.mode,
        "domain": meta.domain.value, "task": meta.task.value,
        "frequency": meta.frequency.value,
        "score": metrics.get("score", 0.0),
        "metrics": metrics,
        "submitted_ref": req.model_ref or "local-registry",
    }
    if req.mode != "private":
        LB.add(record)
    return _to_native(record)


@app.get("/api/leaderboard")
def leaderboard(task: Optional[str] = None, domain: Optional[str] = None):
    rows = LB.query(task=task, domain=domain)
    # re-rank
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return _to_native({"total": len(rows), "rows": rows})


@app.get("/api/leaderboard/{model}")
def leaderboard_model(model: str):
    rows = [r for r in LB.records if r.get("model") == model]
    rows = sorted(rows, key=lambda r: r.get("score", 0), reverse=True)
    return _to_native({"model": model, "rows": rows})


# --------------------------------------------------------------------------- #
# Serve the static frontend (local demo = one server)
# --------------------------------------------------------------------------- #
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)