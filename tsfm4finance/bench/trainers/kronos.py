"""Kronos trainer: drives Kronos' own fine-tuning stack (tokenizer + predictor).

Kronos is trained as an autoregressive model over quantised K-line tokens; its
repository ships the only maintained training loop (``finetune_csv/``).  Rebuilding
that (codebook, tokenizer stage, curriculum) here would produce a model that is no
longer Kronos, so this trainer (a) exports the arena train split into the CSV
layout their config expects, (b) writes a ready-to-run config, and (c) either runs
their trainer (``--run``) or prints the exact command for the GPU server.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import REPO, TrainJob

KRONOS = REPO / "third_party" / "Kronos"


def prepare(job: TrainJob) -> dict:
    """Train-split OHLCV -> ``finetune_csv/data/<series>.csv`` (their schema)."""
    from tsfm4finance.bench.download.common import load_dataset
    tr, _va, _te, meta = load_dataset(job.series)
    cols = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in tr.columns]
    out_dir = KRONOS / "finetune_csv" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{job.series}.csv"
    df = tr.copy()
    df.index.name = "timestamps"
    df.reset_index()[["timestamps"] + cols].to_csv(csv_path, index=False)
    cfg_dir = KRONOS / "finetune_csv" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{job.model_key}__{job.series}__{job.budget}.json"
    epochs = 1 if job.smoke else max(1, job.effective_epochs())
    from tsfm4finance.bench.weights import resolve as _resolve
    from tsfm4finance.bench.config import load_grid
    spec = load_grid().models().get(job.model_key)
    model_id = (spec.weights if spec and spec.weights else job.model_key)
    cfg = {
        "data_path": f"data/{job.series}.csv",
        "train_ratio": 1.0,                     # the arena already split the data
        "val_ratio": 0.0, "test_ratio": 0.0,
        "seq_len": min(job.ctx, 512), "predict_len": job.horizon,
        "stride": max(1, job.bs),
        "d_model": 256, "d_ff": 512, "n_layers": 4, "n_heads": 4,
        "patch_len": 16, "stride_patch": 8,
        "tokenizer_ckpt": _resolve("NeoQuasar/Kronos-Tokenizer-base"),
        "init_from": _resolve(model_id),         # arena checkpoint (Hub id or local dir)
        "batch_size": job.bs, "lr": job.effective_lr(), "epochs": epochs,
        "use_amp": job.amp in ("bf16", "fp16"), "seed": job.seed,
        "save_dir": str(Path(job.out_dir) / f"{job.model_key}__{job.series}__{job.budget}_kronos"),
    }
    cfg_path.write_text(json.dumps(cfg, indent=1))
    return {"csv": str(csv_path), "config": str(cfg_path), "columns": cols,
            "rows": int(len(df)), "ohlcv": bool(meta.get("ohlcv")),
            "note": ("open/high/low/close/volume required; "
                     f"dataset has {cols}")}


def train(job: TrainJob) -> dict:
    prep = prepare(job)
    if not prep["ohlcv"]:
        raise ValueError(f"{job.series} has no OHLCV columns; Kronos needs candlesticks")
    cmd = ["python", "finetune_csv/train_sequential.py",
           "--config", str(Path(prep["config"]).relative_to(KRONOS))]
    if not job.run_training:
        return {"model": job.model_key, "series": job.series, "budget": job.budget,
                "status": "command-prepared", "command": " ".join(cmd),
                "cwd": str(KRONOS), **prep}
    rc = subprocess.call(cmd, cwd=str(KRONOS))
    return {"model": job.model_key, "series": job.series, "budget": job.budget,
            "status": "ran" if rc == 0 else "failed", "returncode": rc,
            "command": " ".join(cmd), "cwd": str(KRONOS), **prep}
