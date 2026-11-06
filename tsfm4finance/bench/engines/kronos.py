"""Kronos (MIT) - the only TSFM in this benchmark pretrained on financial
K-lines.  It consumes OHLCV(+amount) and emits OHLCV, so it needs panel windows
(``Window.panel``), not a single close series; the runner supplies them.

Sampling is part of the model (temperature + nucleus sampling over hierarchical
tokens), so the interval comes from ``sample_count`` paths, and the reported
uncertainty is model sampling uncertainty - recorded in Pred.meta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Engine, Pred, quantiles_from_samples

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "third_party" / "Kronos") not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "Kronos"))


class KronosEngine(Engine):
    key, family = "kronos-base", "kronos"
    caps = {"forecast", "panel_in", "finetune", "ohlcv_out"}
    needs = ("open", "high", "low", "close", "volume")
    max_context = 512
    max_horizon = 512

    def __init__(self, checkpoint: str = "NeoQuasar/Kronos-base",
                 tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
                 sample_count: int = 32, temperature: float = 1.0, top_p: float = 0.9,
                 key: str | None = None, **kw):
        super().__init__(**kw)
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
        self.checkpoint, self.tokenizer_id = checkpoint, tokenizer
        tok = KronosTokenizer.from_pretrained(tokenizer)
        mod = Kronos.from_pretrained(checkpoint)
        self.module, self.tokenizer = mod, tok
        self.predictor = KronosPredictor(mod, tok, device=self.device,
                                         max_context=self.max_context)
        self.sample_count, self.temperature, self.top_p = int(sample_count), float(temperature), float(top_p)
        self.key = key or checkpoint.split("/")[-1]

    def _panels(self, contexts, panels):
        if panels is None:
            raise ValueError("Kronos needs OHLCV panels; run the cell with with_panel=True")
        return [p.tail(self.max_context) for p in panels]

    @staticmethod
    def _cols(df: pd.DataFrame) -> list[str]:
        return [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        ps = self._panels(contexts, panels)
        pts, paths = [], []
        for p in ps:
            cols = self._cols(p)
            x_df = p[cols]
            x_ts = pd.Series(pd.to_datetime(p["timestamp"]))
            last = pd.to_datetime(p["timestamp"]).iloc[-1]
            freq = getattr(p["timestamp"].dropna(), "freq", None)
            try:
                y_ts = pd.Series(pd.date_range(last, periods=horizon + 1, freq=freq or "D")[1:])
            except Exception:                                          # noqa: BLE001
                y_ts = pd.Series(pd.date_range(last, periods=horizon + 1)[1:])
            out = self.predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                         pred_len=horizon, T=self.temperature,
                                         top_p=self.top_p, sample_count=self.sample_count,
                                         verbose=False)
            close = np.asarray(out["close"], dtype=float)
            pts.append(close)
            if self.sample_count > 1:
                paths.append(np.asarray(out.get("close_paths", close), dtype=float)
                             if isinstance(out, dict) else close)
        point = np.stack(pts)[:, :horizon]
        meta = {"sampling": {"T": self.temperature, "top_p": self.top_p,
                             "sample_count": self.sample_count},
                "note": "Kronos emits OHLCV; the benchmark scores the close column"}
        if len(paths) == len(pts) and np.asarray(paths[0]).ndim == 2:
            return Pred(point=point, quantiles=quantiles_from_samples(np.stack(paths, 1)),
                        samples=np.stack(paths, 1), meta=meta)
        return Pred(point=point, meta=meta)

    def trainable(self):
        return self.module, self.tokenizer
