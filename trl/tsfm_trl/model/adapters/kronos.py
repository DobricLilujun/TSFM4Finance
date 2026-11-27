"""Kronos (MIT) — the TSFM pretrained on financial K-lines.

It consumes OHLCV(+amount) panels and emits OHLCV, so its adapter takes
``panels=`` (a DataFrame window per series), not a single close array; the benchmark
scores the ``close`` column and says so in ``Pred.meta``. Sampling (temperature +
nucleus over hierarchical tokens) is part of the model, so the interval is model
sampling uncertainty.

Fine-tuning runs the vendor's own objective in-process (``training_step`` =
next-token cross-entropy through the frozen tokenizer, the same loss its
``finetune_base_model`` optimises) so LoRA, the memory ladder and the loss curve
work here as everywhere. ``vendor_fit`` remains for anyone who wants the vendor's
script instead: it exports the CSV + config and prints the command.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import CapabilityError, UnavailableError
from ..base import Pred, TSFMModel, quantiles_from_samples

COLS = ("open", "high", "low", "close", "volume", "amount")


def _import_kronos():
    """Kronos is a git repo (no wheel): try the pip name, then the repo's own
    ``model.kronos`` after putting the third-party checkouts on the path."""
    from ...thirdparty import ensure_on_path, hint
    ensure_on_path()
    try:
        from kronos.model.kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401
        return Kronos, KronosTokenizer, KronosPredictor, "kronos.model.kronos"
    except Exception:                                            # noqa: BLE001
        pass
    try:
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401
        return Kronos, KronosTokenizer, KronosPredictor, "model.kronos"
    except Exception as e:                                       # noqa: BLE001
        raise UnavailableError("kronos", f"kronos package not importable: {e}",
                               f"git clone https://github.com/shiyu-coder/Kronos ; {hint('Kronos')}") from e


class Kronos(TSFMModel):
    key, family = "kronos-small", "kronos"
    caps = frozenset({"forecast", "panel_in", "paths", "finetune"})
    loss_kind = "token"        # next-token CE; see training_step
    needs = ("open", "high", "low", "close", "volume")
    max_context = 512
    max_horizon = 512
    max_draws = 8            # sampler draws kept for the interval (the point is their mean)

    def __init__(self, checkpoint: str = "NeoQuasar/Kronos-small",
                 tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
                 sample_count: int = 32, temperature: float = 1.0, top_p: float = 0.9, **kw):
        super().__init__(**kw)
        Kronos_, KronosTokenizer, KronosPredictor, self._import_path = _import_kronos()
        self.checkpoint, self.tokenizer_id = checkpoint, tokenizer
        self.tok = KronosTokenizer.from_pretrained(tokenizer)
        self.module = Kronos_.from_pretrained(checkpoint)
        self.predictor = KronosPredictor(self.module, self.tok, device=self.device,
                                         max_context=self.max_context)
        self.sample_count = int(sample_count)
        self.temperature, self.top_p = float(temperature), float(top_p)

    @staticmethod
    def _cols(df: pd.DataFrame) -> list[str]:
        return [c for c in COLS if c in df.columns]

    @staticmethod
    def _flat_panel(series) -> pd.DataFrame:
        """A 1-D series as an OHLCV panel: open=high=low=close, volume=0.

        This is not a shortcut invented here — it is exactly how the vendor's own
        ``finetune_csv`` format represents a single price series (its CSV writes the
        same value into every price column), and Kronos trains and infers on that.
        Real panels (``panels=``) always win when supplied.
        """
        v = np.asarray(series, float)
        return pd.DataFrame({"open": v, "high": v, "low": v, "close": v,
                             "volume": np.zeros_like(v),
                             "amount": np.zeros_like(v),
                             "timestamp": pd.date_range("2020-01-01", periods=len(v),
                                                        freq="D")})

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        if panels is None:
            # every downstream caller (probe features, anomaly, reward) hands us 1-D
            # windows; refusing them would leave three matrix columns blank for a model
            # that happily trains on flat panels
            panels = [self._flat_panel(c) for c in contexts]
        self._last_panels = panels
        horizon = int(horizon)
        # Kronos' own sampler averages itself away: ``predict`` averages
        # ``sample_count`` decodes into one frame, so the spread it drew is not
        # retrievable — and the spread is the only honest interval a sampler has.
        # The draws are batched (series × draws goes through ``predict_batch`` once),
        # because 32 sequential decodes per series turned a matrix run into an hour.
        draws = max(1, min(int(self.sample_count), int(self.max_draws)))
        dfs, ts_list, ys_list = [], [], []
        for _ in range(draws):
            for pn in panels:
                df = pn.tail(self.max_context)
                ts = (pd.Series(pd.to_datetime(df["timestamp"]))
                      if "timestamp" in df.columns else pd.Series(pd.RangeIndex(len(df))))
                last = ts.iloc[-1]
                try:
                    freq = getattr(pd.DatetimeIndex(ts).dropna(), "freq", None)
                    y_ts = pd.Series(pd.date_range(last, periods=horizon, freq=freq or "D"))
                except Exception:                                # noqa: BLE001
                    y_ts = pd.Series(np.arange(len(ts), len(ts) + horizon))
                dfs.append(df[self._cols(df)])
                ts_list.append(ts)
                ys_list.append(y_ts)
        frames = self.predictor.predict_batch(dfs, ts_list, ys_list, pred_len=horizon,
                                              T=self.temperature, top_p=self.top_p,
                                              sample_count=1, verbose=False)
        cols = []
        for fr in frames:
            frame = fr if isinstance(fr, pd.DataFrame) else pd.DataFrame(np.asarray(fr, float))
            v = (frame["close"].to_numpy() if "close" in getattr(frame, "columns", [])
                 else np.asarray(fr, float).reshape(-1))
            cols.append(np.asarray(v, float).reshape(-1)[:horizon])
        S = np.stack(cols).reshape(draws, len(panels), horizon).transpose(1, 0, 2)  # (B,K,H)
        P = S.mean(axis=1)
        meta = {"sampling": {"T": self.temperature, "top_p": self.top_p, "draws": draws},
                "note": "Kronos emits OHLCV; close is what is scored, and the interval is "
                        "the model's own sampler spread (it has no quantile head)"}
        return Pred(point=P, quantiles=quantiles_from_samples(S), samples=S, meta=meta)

    def sample_paths(self, contexts, horizon, n=8):
        if hasattr(self, "_last_panels") and self._last_panels is not None:
            p = self.predict(contexts, horizon, panels=self._last_panels)
            if p.samples is not None:
                return np.asarray(p.samples, float)[:, :n, :horizon]
        raise CapabilityError(self.key, "paths", "Kronos paths need OHLCV panels")

    def trainable(self):
        return self.module

    # ------------------------------------------------------------------ fine-tune
    def prepare_training(self, ctx: int, horizon: int) -> None:
        self._train_window = min(int(ctx) + int(horizon), self.max_context)

    def _train_tensors(self, seqs):
        """Normalise + stamp windows exactly as the vendor's finetune dataset does
        (per-window mean/std, clip, five calendar time features)."""
        import importlib
        calc = getattr(importlib.import_module(self._import_path), "calc_time_stamps")
        X, ST = [], []
        for seq in seqs:
            df = self._flat_panel(seq)
            # the model's input width is the vendor's feature list: price columns +
            # volume + amount. An all-equal OHLC panel has no real volume/amount, so
            # they stay zero — but they must still be *there* or the embedding matmul
            # is a shape error.
            cols = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df]
            x = df[cols].values.astype(np.float32)
            st = calc(pd.Series(pd.DatetimeIndex(df["timestamp"]))).values.astype(np.float32)
            x = np.clip((x - x.mean(0)) / (x.std(0) + 1e-5), -5.0, 5.0)
            X.append(x)
            ST.append(st)
        import torch
        return torch.tensor(np.stack(X)), torch.tensor(np.stack(ST))

    def training_step(self, contexts, truth, horizon):
        """Next-token cross-entropy over the hierarchical tokens — the objective the
        vendor's ``finetune_base_model`` optimises (tokenizer.encode, then
        ``head.compute_loss``), run in-process so the shared trainer, LoRA and the loss
        curve apply to Kronos like every other family. The tokenizer stays frozen; the
        transformer is what trains."""
        import torch
        seqs = [np.r_[np.asarray(c, float).reshape(-1), np.asarray(t, float).reshape(-1)]
                for c, t in zip(contexts, truth)]
        n = min(len(q) for q in seqs)
        seqs = [q[-n:] for q in seqs]
        x, st = self._train_tensors(seqs)
        x, st = x.to(self.device), st.to(self.device)
        self.tok.to(self.device)
        with torch.no_grad():
            t0, t1 = self.tok.encode(x, half=True)
        logits = self.module(t0[:, :-1], t1[:, :-1], st[:, :-1, :])
        loss, _, _ = self.module.head.compute_loss(logits[0], logits[1], t0[:, 1:], t1[:, 1:])
        return loss

    # ------------------------------------------------------------------ vendor fit
    def vendor_fit(self, windows, cfg):
        """Export the vendor's CSV format, then run (or print) its finetune script."""
        if not windows:
            raise ValueError("no windows")
        work = Path(cfg.out_dir).expanduser() / f"{self.key}__kronos"
        work.mkdir(parents=True, exist_ok=True)
        rows = []
        for c, y in windows:
            seq = np.r_[c, y]
            rows += [{"time_id": i, "open": v, "high": v, "low": v, "close": v, "volume": 0.0}
                     for i, v in enumerate(seq)]
        csv = work / "train.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        conf = {"data_path": str(csv), "config": {"criteria_path": None}}
        (work / "config.json").write_text(json.dumps(conf, indent=1))
        cmd = [sys.executable, "-m", "kronos.finetune.finetune_utils", "--config", str(work)]
        if not (cfg.extra or {}).get("run_training"):
            return {"status": "prepared", "reason": "vendor fine-tune prepared but not run "
                    "(pass extra={'run_training': True}); CSV at " + str(csv),
                    "command": " ".join(cmd), "csv": str(csv), "steps": 0}
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.extra.get("timeout", 7200))
        return {"status": "ok" if out.returncode == 0 else "failed",
                "command": " ".join(cmd), "steps": 0,
                "reason": (out.stderr or "")[-400:]}
