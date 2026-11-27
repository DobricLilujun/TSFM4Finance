"""Shared plumbing for the per-family trainers in this package."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FT_DIR = REPO / "reports" / "ft_ckpts"

# Budget semantics are part of the benchmark definition, not a knob to hide:
# fs16/fs64 are the few-shot adaptation points, full is the unconstrained one.
BUDGETS = {
    "fs16": {"n_windows": 16, "steps": 40, "epochs": 0, "lr": 1e-5},
    "fs64": {"n_windows": 64, "steps": 120, "epochs": 0, "lr": 1e-5},
    "full": {"n_windows": None, "steps": None, "epochs": 3, "lr": 5e-6},
}
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class TrainJob:
    model_key: str
    series: str
    budget: str = "full"
    ctx: int = 512
    horizon: int = 21
    amp: str = "fp32"                  # fp32 | bf16 | fp16
    bs: int = 4
    lr: float | None = None
    epochs: int | None = None
    device: str = "cuda"
    seed: int = 0
    smoke: bool = False
    heads_only: bool = False
    run_training: bool = False       # vendor-stack trainers (Kronos) default to printing the command
    out_dir: str = str(FT_DIR)

    def budget_kw(self) -> dict:
        return BUDGETS[self.budget]

    def ckpt_path(self) -> Path:
        d = Path(self.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.model_key}__{self.series}__{self.budget}.pt"

    def effective_lr(self) -> float:
        return float(self.lr or self.budget_kw()["lr"])

    def effective_epochs(self) -> int:
        return int(self.epochs if self.epochs is not None else self.budget_kw()["epochs"])


# ------------------------------------------------------------------ data windows
def train_windows(series: str, horizon: int, ctx: int, seed: int = 0,
                  n_windows: int | None = None, target: str = "close"):
    """Supervised windows taken **strictly inside the train split**.

    This is what makes the fine-tune honest: validation and test rows are never
    seen, and ``info['train_end']`` is recorded in the checkpoint so the sweep's
    validation-span control can prove it.
    """
    import pandas as pd
    from tsfm4finance.bench.download.common import load_dataset
    tr, va, te, meta = load_dataset(series)
    target = str(meta.get("target", target))
    y = tr[target].to_numpy(dtype=float)
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    ctx_eff = int(ctx)
    for start in range(ctx_eff, len(y) - horizon):
        c = y[start - ctx_eff:start]
        t = y[start:start + horizon]
        if len(c) != ctx_eff or len(t) < horizon:      # equal-length batches: the
            continue                                   # trainers stack them directly
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(t))):
            continue
        windows.append((c.astype(np.float32), t.astype(np.float32)))
    rng = np.random.RandomState(seed)
    if n_windows and len(windows) > n_windows:
        idx = rng.choice(len(windows), size=n_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]
    info = {"series": series, "target": target, "n_train_obs": int(len(y)),
            "train_end": str(tr.index[-1]) if len(tr) else None,
            "n_windows": len(windows), "horizon": int(horizon), "ctx": int(ctx),
            "ohlcv": bool(meta.get("ohlcv")), "panel_cols": [c for c in tr.columns
                                                             if c != target]}
    return windows, info


def batch_arrays(windows, bs: int, rng: np.random.RandomState):
    """Random minibatch of (context, truth)."""
    idx = rng.choice(len(windows), size=min(bs, len(windows)), replace=False)
    ctxs = np.stack([windows[i][0] for i in idx])
    truth = np.stack([windows[i][1] for i in idx])
    return ctxs, truth


def panel_batch(windows, bs: int, rng, cols: list[str], series: str):
    """OHLCV candle batches for K-line models (Kronos)."""
    import pandas as pd
    from tsfm4finance.bench.download.common import load_dataset
    tr, _va, _te, _meta = load_dataset(series)
    full = tr[[c for c in cols if c in tr.columns]]
    idx = rng.choice(max(1, len(full) - 1), size=min(bs, max(1, len(full) - 1)), replace=False)
    out = []
    for i in idx:
        lo = max(0, i - 512)
        out.append(full.iloc[lo:i].to_numpy(dtype=float))
    return out


# ------------------------------------------------------------------ loss + misc
def pinball_loss_t(q: "torch.Tensor", y: "torch.Tensor", levels=LEVELS):
    """Pinball loss over ``q`` (B,H,nq) vs ``y`` (B,H); mean over everything."""
    import torch
    lv = torch.tensor(levels, device=q.device, dtype=q.dtype)
    y = y.unsqueeze(-1)
    return torch.maximum(lv * (y - q), (lv - 1.0) * (y - q)).mean()


def combined_loss_t(q, y, levels=LEVELS, point_index: int | None = None, w_point: float = 1.0):
    """Pinball over the quantile grid + L1 on the median channel.

    The extra L1 term matters: pure pinball lets the median drift between grid
    levels on short financial windows, which then shows up as a directional bias
    in the trading task.
    """
    import torch
    if q is None:
        return None
    med = int(levels.index(0.5)) if 0.5 in levels else (point_index or q.shape[-1] // 2)
    return pinball_loss_t(q, y, levels) + w_point * torch.abs(q[..., med] - y).mean()


def autocast(device: str, amp: str):
    import torch
    if device != "cuda" or amp not in ("bf16", "fp16"):
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def count_trainable(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def set_trainable(module, heads_only: bool, head_prefixes: tuple[str, ...]) -> list[str]:
    """``--heads-only`` ablation switch; the names actually trained are returned
    and stored in metadata, so the paper can never mislabel a run."""
    if not heads_only:
        module.requires_grad_(True)
        return ["*all*"]
    module.requires_grad_(False)
    touched = []
    for name, p in module.named_parameters():
        if any(name.startswith(h) for h in head_prefixes):
            p.requires_grad_(True)
            touched.append(name)
    return touched


def save_ckpt(job: TrainJob, module, meta: dict, extra_state: dict | None = None) -> str:
    import torch
    path = job.ckpt_path()
    state = {"model": module.state_dict() if hasattr(module, "state_dict") else module,
             "meta": meta}
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return str(path)


def load_ckpt(path: str | Path, map_location: str = "cpu"):
    import torch
    return torch.load(str(path), map_location=map_location, weights_only=False)


def log_step(hist: dict, step: int, loss: float, t0: float) -> None:
    hist.setdefault("steps", []).append([int(step), round(float(loss), 5)])
    if step % 10 == 0 or step == 1:
        print(json.dumps({"step": step, "loss": round(float(loss), 5),
                          "sec": round(time.time() - t0, 1)}), flush=True)


def free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                              # noqa: BLE001
        pass
