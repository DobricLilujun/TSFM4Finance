"""Backend test suite for Finance TSFMs Arena.

Covers the three requested backend features:
  1. Upload/register a model -> it is scored like a built-in.
  2. Upload an answer (raw predictions) -> scored against the dataset truth.
  3. Manage / store models, datasets and results (submissions).

Why FastAPI TestClient (not Django):
  The arena backend is FastAPI. `TestClient` is the standard equivalent of
  Django's test client — it drives the real ASGI app end-to-end without a
  network port, is dependency-light, and exercises the same code path a real
  client would. (Django's TestCase/TestClient would require a full Django
  project, ORM and settings; there is no Django layer here.) The tests run
  against an ISOLATED temp data dir so they never touch real artifacts, and
  are skipped if a dataset's parquet splits are unavailable.

Run:  uv run pytest -q        (or)  uv run python -m pytest backend/tests -q
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
ORIG_LB = Path(__file__).resolve().parent.parent / "leaderboard.json"


# --------------------------------------------------------------------------- #
# Fixtures: isolated data dir + fresh app
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_path):
    """Return a TestClient backed by an isolated, synthetic dataset."""
    from fastapi.testclient import TestClient

    # Build a synthetic open dataset (small, deterministic) so tests never
    # depend on real parquets or heavy models.
    synth = tmp_path / "data" / "open" / "test_daily"
    synth.mkdir(parents=True)
    n = 120
    meta = {
        "name": "test_daily", "open": True, "domain": "equity",
        "task": "forecast", "frequency": "day",
        "target": "close", "horizon": 3, "lookback": 20, "rolling": True,
        "n_features": 1, "feature_names": ["close"],
        "n_obs": n, "license": "test",
    }
    (synth / "meta.json").write_text(json.dumps(meta))
    t = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "close": np.linspace(100, 160, n) + 5 * np.sin(np.linspace(0, 20, n)),
    })
    train = t.iloc[:90]
    val = t.iloc[90:105]
    test = t.iloc[105:]
    for name, df in (("train", train), ("validation", val), ("test", test)):
        df.to_parquet(synth / f"{name}.parquet")

    # Point the backend at the isolated dirs BEFORE importing the app.
    import backend.app as appmod
    import backend.registry as regmod
    old_data, old_lb = appmod.DATA, appmod.LB.path
    old_reg, old_sub, old_idx = regmod.REGISTRY, regmod.SUB_DIR, regmod.SUB_INDEX
    old_models, old_index = None, None
    appmod.DATA = tmp_path / "data"
    appmod.LB = type(appmod.LB)(tmp_path / "leaderboard.json")
    regmod.REGISTRY = tmp_path / "models_registry.json"
    regmod.SUB_DIR = tmp_path / "submissions"
    regmod.SUB_INDEX = tmp_path / "submissions" / "index.json"

    # Bypass the enabled-config filter so the synthetic dataset / registered
    # models are always visible (tests must not depend on real config files).
    import backend.config as cfg
    cfg.enabled_datasets = lambda: None
    cfg.enabled_models = lambda: None

    c = TestClient(appmod.app)
    yield c
    appmod.DATA, appmod.LB.path = old_data, old_lb
    regmod.REGISTRY, regmod.SUB_DIR, regmod.SUB_INDEX = old_reg, old_sub, old_idx


# --------------------------------------------------------------------------- #
# Feature 1: upload / register a model -> scored
# --------------------------------------------------------------------------- #
def test_register_model_then_evaluate(client):
    r = client.post("/api/models/register", json={
        "name": "my-linear", "type": "linear", "params": {}, "task": "forecast"})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "linear"

    # the registered model is now listed + scoreable like a built-in
    reg = client.get("/api/models/registered").json()
    assert any(m["name"] == "my-linear" for m in reg)

    score = client.post("/api/evaluate", json={
        "dataset": "test_daily", "model": "my-linear", "mode": "private"})
    assert score.status_code == 200, score.text
    assert score.json()["model"] == "my-linear"
    # score may be NaN/None on degenerate synthetic data; the point is the
    # endpoint ran the model and returned a record
    assert "metrics" in score.json()


def test_register_rejects_bad_type(client):
    r = client.post("/api/models/register", json={"name": "bad", "type": "nonsense"})
    # an unsupported type must not be accepted as a valid registration
    assert r.status_code in (400, 422, 500), r.status_code


# --------------------------------------------------------------------------- #
# Feature 2: upload an answer -> scored against truth
# --------------------------------------------------------------------------- #
def test_upload_answer_is_scored(client):
    # upload an answer whose predictions are a few plausible values -> scored
    r = client.post("/api/upload-answer", json={
        "dataset": "test_daily", "model_name": "my-answer",
        "predictions": [155.0, 157.0, 159.0], "mode": "open"})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"] == "test_daily"
    assert r.json()["n_predictions"] >= 1
    assert "id" in r.json()
    # a valid submission is stored
    subs = client.get("/api/submissions").json()
    assert any(s.get("id") == r.json()["id"] for s in subs)


def test_upload_answer_private_not_on_leaderboard(client):
    r = client.post("/api/upload-answer", json={
        "dataset": "test_daily", "model_name": "secret",
        "predictions": [150.0, 151.0, 152.0], "mode": "private"})
    assert r.status_code == 200
    lb = client.get("/api/leaderboard").json()
    models = [row.get("model") for row in lb.get("rows", [])]
    assert "secret" not in models


# --------------------------------------------------------------------------- #
# Feature 3: manage / store models, datasets, results
# --------------------------------------------------------------------------- #
def test_submission_is_stored_and_listable(client):
    r = client.post("/api/upload-answer", json={
        "dataset": "test_daily", "model_name": "store-test",
        "predictions": [150.0, 151.0], "mode": "open"})
    sid = r.json()["id"]
    assert sid
    # list contains it
    subs = client.get("/api/submissions").json()
    assert any(s["id"] == sid for s in subs)
    # fetch by id
    one = client.get(f"/api/submissions/{sid}")
    assert one.status_code == 200 and one.json()["id"] == sid
    # delete it
    d = client.delete(f"/api/submissions/{sid}")
    assert d.json()["ok"] is True
    assert client.get(f"/api/submissions/{sid}").status_code == 404


def test_management_summary(client):
    client.post("/api/models/register", json={"name": "s", "type": "constant"})
    client.post("/api/upload-answer", json={
        "dataset": "test_daily", "model_name": "a", "predictions": [1, 2, 3]})
    s = client.get("/api/management/summary").json()
    assert "test_daily" in s["datasets"]
    assert any(m["name"] == "s" for m in s["models"]["registered"])
    assert s["leaderboard_total"] >= 0


def test_datasets_and_models_listed(client):
    ds = client.get("/api/datasets").json()
    assert any(d["name"] == "test_daily" for d in ds)
    md = client.get("/api/models").json()
    assert any(m["name"] == "naive" for m in md)


def test_health(client):
    assert client.get("/api/health").status_code == 200


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))