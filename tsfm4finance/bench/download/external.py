"""T2 external benchmark sets, so our numbers are comparable to published ones.

* fev-bench (`autogluon/fev_datasets`): `fred_md_2025`, `fred_qd_2025`,
  `us_consumption` -- econ/finance panels, including covariate columns (the `x`
  suffix convention), which is what lets us exercise the covariate-capable TSFMs
  (Chronos-2, TimesFM 3.0, Toto) rather than only univariate ones.
* GIFT-Eval (`Salesforce/GiftEval`): M4 daily/weekly/hourly slices, sampled by
  length, purely as a protocol cross-check -- if our rolling harness reproduces
  published MASE ranges on data the leaderboard uses, the harness is credible.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from .common import PANELS, dataset_exists, record_failure, clear_failure, write_open_dataset

FEV_TASKS = ["fred_md_2025", "fred_qd_2025", "us_consumption"]
GIFT_TASKS = {"m4_daily": "day", "m4_weekly": "week", "m4_hourly": "hour"}


def _fev_parquet(task: str) -> pd.DataFrame:
    p = hf_hub_download(repo_id="autogluon/fev_datasets", subfolder=task,
                        filename="train-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(p)


def import_fev(task: str, *, max_variates: int = 40, min_obs: int = 400,
               force: bool = False) -> dict:
    """fev rows are {id, timestamp:[...], variate:[...], covariate:[...]x}.

    The wide panel is always stored (multivariate models read it); individual
    univariate datasets are written for the ``max_variates`` most volatile
    variates so the per-series sweep stays tractable.
    """
    raw = _fev_parquet(task)
    row = raw.iloc[0]
    ts = pd.to_datetime(pd.Series(row["timestamp"]))
    cols = [c for c in raw.columns if c not in ("id", "timestamp")]
    wide = {}
    for c in cols:
        v = np.asarray(row[c], dtype=float)
        if len(v) == len(ts):
            wide[c] = v
    panel = pd.DataFrame(wide, index=ts).sort_index()
    panel.index.name = "timestamp"
    PANELS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANELS / f"{task}.parquet")
    (PANELS / f"{task}_meta.json").write_text(json.dumps({
        "task": task, "n_variates": int(panel.shape[1]),
        "covariate_columns": [c for c in panel.columns if c.endswith("x")],
        "date_range": [str(panel.index.min())[:10], str(panel.index.max())[:10]],
        "format": "wide parquet; columns ending in 'x' are fev covariates",
    }, indent=1))

    target_cols = [c for c in panel.columns if not c.endswith("x")]
    if not target_cols:
        target_cols = list(panel.columns)
    vol = panel[target_cols].std()
    picks = [c for c in target_cols
             if np.isfinite(vol[c]) and panel[c].notna().sum() >= min_obs]
    picks = sorted(picks, key=lambda c: -vol[c])[:max_variates]
    res = {"_panel": "ok"}
    for c in picks:
        name = f"fev_{task}_{c}".replace("/", "_").replace(".", "_")[:64]
        if dataset_exists(name) and not force:
            res[c] = "exists"
            continue
        s = panel[[c]].dropna().reset_index().rename(columns={c: "close"})
        if len(s) < min_obs:
            continue
        write_open_dataset(name, s, freq={"fred_md_2025": "day", "fred_qd_2025": "quarter",
                                          "us_consumption": "week"}.get(task, "day"),
                           domain="macro", source=f"fev-bench:{task}:{c}",
                           license_="see fev-bench dataset terms",
                           notes="fev-bench variate imported verbatim (normalised "
                                 "levels as published by fev)",
                           extra_meta={"fev_task": task, "variate": c})
        res[c] = "ok"
    return res


def import_gift(task: str, *, n_series: int = 40, min_len: int = 700,
                force: bool = False) -> dict:
    """GIFT-Eval arrow shard -> per-series datasets (protocol cross-check set)."""
    fname = "data-00000-of-00001.arrow"
    p = hf_hub_download(repo_id="Salesforce/GiftEval", subfolder=task,
                        filename=fname, repo_type="dataset")
    ds = load_dataset("arrow", data_files=p, split="train")
    picks, seen = [], 0
    for r in ds:
        tgt = np.asarray(r["target"], dtype=float)
        if tgt.size >= min_len and np.isfinite(tgt).all() and tgt.std() > 0:
            picks.append((str(r["item_id"]), tgt, pd.to_datetime(r["start"])))
            seen += 1
            if seen >= n_series:
                break
    res = {}
    for item_id, tgt, start in picks:
        name = f"ge_{task}_{item_id}"[:64]
        if dataset_exists(name) and not force:
            res[item_id] = "exists"
            continue
        idx = pd.date_range(start, periods=len(tgt),
                            freq={"m4_daily": "D", "m4_weekly": "W",
                                  "m4_hourly": "h"}[task])
        df = pd.DataFrame({"timestamp": idx, "close": tgt}).dropna()
        write_open_dataset(name, df, freq=GIFT_TASKS[task], domain="econ_fin_m4",
                           source=f"gift-eval:{task}:{item_id}",
                           license_="CC-BY (GIFT-Eval redistribution of M4)",
                           notes="GIFT-Eval series used only to cross-check the "
                                 "harness against published leaderboard MASE ranges",
                           extra_meta={"gift_task": task, "item_id": item_id})
        res[item_id] = "ok"
    if not res:
        record_failure("external", task, "no usable series")
    else:
        clear_failure("external", task)
    return res


def run(force: bool = False) -> dict:
    out: dict[str, dict] = {}
    for t in FEV_TASKS:
        try:
            out[f"fev:{t}"] = import_fev(t, force=force)
        except Exception as e:                                  # noqa: BLE001
            record_failure("external", f"fev:{t}", str(e))
            out[f"fev:{t}"] = {"error": str(e)[:200]}
    for t in GIFT_TASKS:
        try:
            out[f"gift:{t}"] = import_gift(t, force=force)
        except Exception as e:                                  # noqa: BLE001
            record_failure("external", f"gift:{t}", str(e))
            out[f"gift:{t}"] = {"error": str(e)[:200]}
    return out
