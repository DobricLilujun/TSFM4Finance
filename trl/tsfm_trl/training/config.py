"""Training configuration.

Budgets are part of the *definition* of an adaptation run, not a knob to lose in a
CLI: ``fs16`` (16 windows / 40 steps) and ``fs64`` (64 / 120) are the few-shot
points, ``full`` is the unconstrained one. A checkpoint records which one ran.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUDGETS = {
    "fs16": {"n_windows": 16, "steps": 40, "epochs": 0, "lr": 1e-5},
    "fs64": {"n_windows": 64, "steps": 120, "epochs": 0, "lr": 1e-5},
    "full": {"n_windows": None, "steps": None, "epochs": 3, "lr": 5e-6},
}

# adapter-name conventions for peft="heads_only": families without an explicit
# head_prefixes declaration fall back to these.
DEFAULT_HEAD_PREFIXES = ("head", "output_proj", "projection", "quantile", "p21",
                         "distribution_output", "fc_out", "lm_head", "decoder")


@dataclass
class TrainingConfig:
    """Everything a trainer needs that is not the model or the data."""

    ctx: int = 256
    horizon: int = 21
    budget: str = "fs16"                 # fs16 | fs64 | full | custom
    split: str = "train"                 # windows are sampled from this split only
    steps: int | None = None             # None -> from budget
    epochs: int | None = None
    n_windows: int | None = None         # few-shot window count (None = all)
    lr: float | None = None
    bs: int = 4
    amp: str = "fp32"                    # fp32 | bf16 | fp16
    device: str = "auto"
    seed: int = 0
    #: GRPO loop size for the matrix (kept tiny on purpose — the reward column is a
    #: training *capability* check, not a training result)
    reward_steps: int = 6
    reward_candidates: int = 4
    peft: str = "full"                   # full | heads_only | lora
    lora_r: int = 8
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj",
                                     "o_proj", "wqkv", "fc1", "fc2")
    grad_clip: float = 1.0
    warmup: int = 0
    weight_decay: float = 0.01
    smoke: bool = False                  # 2 optimiser steps: wiring proof only
    out_dir: str = "reports/ft"
    log_every: int = 10
    tag: str = ""                        # extra suffix for the checkpoint name
    max_seconds: float = 0.0             # 0 = no wall-clock cap
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def __post_init__(self):
        if self.budget not in BUDGETS and self.budget != "custom":
            raise ValueError(f"budget must be one of {sorted(BUDGETS)} or 'custom'")
        # the CLI spells "not given" as 0 (argparse defaults read badly otherwise); those
        # sentinels must not reach the training loop as a batch size of zero
        if not self.bs or self.bs <= 0:
            self.bs = 4
        if not self.lr or self.lr <= 0:
            self.lr = None
        if not self.steps or self.steps <= 0:
            self.steps = None
        if self.n_windows is not None and self.n_windows <= 0:
            self.n_windows = None
        if not self.ctx or self.ctx <= 0:
            self.ctx = 256
        if not self.horizon or self.horizon <= 0:
            self.horizon = 21
        if self.smoke:
            self.steps = 2
            self.out_dir = str(Path(self.out_dir).parent / "ft_smoke")

    @property
    def budget_kw(self) -> dict:
        return BUDGETS.get(self.budget, {"n_windows": None, "steps": None, "epochs": 3, "lr": 1e-5})

    def eff_lr(self) -> float:
        return float(self.lr if self.lr is not None else self.budget_kw["lr"])

    def eff_n_windows(self) -> int | None:
        return self.n_windows if self.n_windows is not None else self.budget_kw["n_windows"]

    def eff_steps(self, n_windows: int) -> int:
        if self.steps:
            return int(self.steps)
        if self.budget_kw["steps"]:
            return int(self.budget_kw["steps"])
        ep = int(self.epochs if self.epochs is not None else self.budget_kw["epochs"] or 3)
        return max(1, ep * max(1, n_windows // max(1, self.bs)))

    def name(self, model_key: str) -> str:
        bits = [model_key, self.budget]
        if self.tag:
            bits.append(self.tag)
        return "__".join(bits)

    def ckpt_path(self, model_key: str) -> Path:
        d = Path(self.out_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name(model_key)}.pt"

    def to_dict(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        d["budget_kw"] = self.budget_kw
        return d


@dataclass
class TrainResult:
    model: str
    status: str                          # ok | smoke | skipped | failed
    steps: int = 0
    final_loss: float = float("nan")
    first_loss: float = float("nan")
    checkpoint: str = ""
    trainable_parameters: int = 0
    trained_parameters: list = field(default_factory=list)
    n_windows: int = 0
    seconds: float = 0.0
    reason: str = ""                     # populated for skipped/failed, never blank
    device: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).with_suffix(".json").write_text(json.dumps(self.to_dict(), indent=1, default=str))


class StepLog:
    """Tiny loss-curve logger: the curve goes into the checkpoint, so a fine-tune
    can be audited without re-running it."""

    def __init__(self, every: int = 10):
        self.steps: list = []
        self.every = every
        self.t0 = time.time()

    def step(self, i: int, loss: float) -> None:
        self.steps.append([int(i), round(float(loss), 6)])
        if i == 1 or i % self.every == 0:
            # stderr, always: these are progress lines, and stdout is a data channel
            # (`doctor --json -` pipes a JSON report; mixing step objects into it breaks
            # the reader, which is how the env-dispatch run lost every row once)
            import sys
            print(json.dumps({"step": i, "loss": round(float(loss), 6),
                              "sec": round(time.time() - self.t0, 1)}),
                  file=sys.stderr, flush=True)

    @property
    def first(self) -> float:
        return self.steps[0][1] if self.steps else float("nan")

    @property
    def last(self) -> float:
        return self.steps[-1][1] if self.steps else float("nan")
