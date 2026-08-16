"""Registry + storage for user-submitted models, answers, and results.

Backs three backend features requested for the arena:
  1. Upload a *model*  -> register it as a runnable adapter, then it can be
     evaluated/scored like any built-in model.
  2. Upload *answers*  -> a set of predictions for a dataset is scored against
     the held-out truth and (optionally) added to the public leaderboard.
  3. *Manage / store*  -> every submission and every registered model is
     persisted (JSON) and can be listed / fetched / removed.

Storage layout (all JSON, human-inspectable, no DB required):
  backend/models_registry.json     registered models (spec + id)
  data/submissions/               one JSON file per submitted answer/result
  data/submissions/index.json     fast lookup index of submissions
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from arena.schemas import DatasetMeta, PredictOutput
from arena.model_base import BaseModel, PredictionResult

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
REGISTRY = HERE / "models_registry.json"
SUB_DIR = DATA / "submissions"
SUB_INDEX = SUB_DIR / "index.json"


# --------------------------------------------------------------------------- #
# Registered models
# --------------------------------------------------------------------------- #
def _load_models() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {}


def _save_models(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def register_model(spec: dict) -> dict:
    """Register a user model. Returns the stored record (with an assigned id).

    Accepted spec keys:
      name, type in {constant, linear, arima}, params (dict), task (optional)
    """
    reg = _load_models()
    name = (spec.get("name") or f"user-{int(time.time())}").strip()
    mtype = spec.get("type", "constant")
    if mtype not in ("constant", "linear", "arima"):
        raise ValueError(f"unsupported model type '{mtype}' (use constant/linear/arima)")
    mid = name + "_" + hashlib.sha1(
        f"{name}{mtype}{json.dumps(spec.get('params', {}))}".encode()
    ).hexdigest()[:8]
    rec = {
        "id": mid, "name": name, "type": mtype,
        "params": spec.get("params", {}) or {},
        "task": spec.get("task", "forecast"),
        "created": time.time(),
    }
    reg[mid] = rec
    _save_models(reg)
    return rec


def get_model(id_or_name: str) -> Optional[dict]:
    reg = _load_models()
    if id_or_name in reg:
        return reg[id_or_name]
    for r in reg.values():
        if r.get("name") == id_or_name:
            return r
    return None


def list_models() -> list[dict]:
    return sorted(_load_models().values(), key=lambda r: r.get("created", 0), reverse=True)


# --------------------------------------------------------------------------- #
# Submissions storage
# --------------------------------------------------------------------------- #
def _load_index() -> dict:
    if SUB_INDEX.exists():
        try:
            return json.loads(SUB_INDEX.read_text())
        except Exception:
            pass
    return {}


def _save_index(idx: dict) -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    SUB_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2))


def store_submission(record: dict) -> str:
    """Persist a submission (answer + score). Returns its id."""
    sid = f"{record.get('dataset','?')}_{record.get('model_name','?')}_" + \
        hashlib.sha1(f"{record.get('dataset')}|{record.get('model_name')}|"
                    f"{time.time()}".encode()).hexdigest()[:10]
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["id"] = sid
    record["created"] = time.time()
    (SUB_DIR / f"{sid}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    idx = _load_index()
    idx[sid] = {k: record[k] for k in
                ("id", "dataset", "model_name", "mode", "score", "created")}
    _save_index(idx)
    return sid


def get_submission(sid: str) -> Optional[dict]:
    p = SUB_DIR / f"{sid}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def list_submissions() -> list[dict]:
    idx = _load_index()
    return sorted(idx.values(), key=lambda r: r.get("created", 0), reverse=True)


def delete_submission(sid: str) -> bool:
    p = SUB_DIR / f"{sid}.json"
    ok = False
    if p.exists():
        p.unlink()
        ok = True
    idx = _load_index()
    if sid in idx:
        del idx[sid]
        _save_index(idx)
        ok = True
    return ok


# --------------------------------------------------------------------------- #
# A BaseModel adapter for registered (user) models
# --------------------------------------------------------------------------- #
class RegisteredModel(BaseModel):
    """Runs a registered user-model spec behind the unified predict() contract.

    Supported types:
      constant -> last observed value (naive)
      linear   -> linear trend fitted on the context window, extrapolated
      arima    -> delegate to the built-in ARIMA adapter
    """

    def __init__(self, spec: dict):
        super().__init__(spec.get("name", "user-model"),
                         supports={spec.get("task", "forecast")})
        self.spec = spec
        self.mtype = spec.get("type", "constant")
        self.params = spec.get("params", {}) or {}

    def predict(self, context_df: pd.DataFrame, meta: DatasetMeta,
                horizon: int | None = None) -> PredictionResult:
        horizon = horizon or meta.horizon or 1
        series = self._target_series(context_df, meta)

        if self.mtype == "arima":
            from arena.models import _build
            try:
                return _build("arima").predict(context_df, meta, horizon=horizon)
            except Exception:
                pass

        if self.mtype == "linear" and len(series) >= 2:
            x = np.arange(len(series), dtype=float)
            slope, intercept = np.polyfit(x, series, 1)
            future_x = np.arange(len(series), len(series) + horizon, dtype=float)
            pts = (slope * future_x + intercept).tolist()
            return PredictionResult(point=pts)

        # constant / default -> naive (last value)
        last = series[-1] if len(series) else 0.0
        return PredictionResult(point=[float(last)] * horizon)

    @staticmethod
    def _target_series(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
        if meta.target in df.columns:
            return df[meta.target].to_numpy(dtype=float)
        numeric = df.select_dtypes(include="number")
        return numeric.iloc[:, -1].to_numpy(dtype=float) if len(numeric) else np.array([])


def make_registered(spec: dict) -> BaseModel:
    return RegisteredModel(spec)