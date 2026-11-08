"""Engine registry: config key -> engine instance, inside the family's own env.

``build_engine`` is the only place that knows which class implements which family,
so adding a model means adding a config row (plus, if needed, a module here).
"""
from __future__ import annotations

from .base import LEVELS, Pred, quantiles_from_samples, to_levels
from ..weights import resolve as _resolve


def build_engine(spec, *, adaptation: str = "zero", ft_ckpt: str | None = None,
                 device: str = "cuda", amp: str = "fp32", batch: int = 16,
                 ctx_cap: int | None = None, seed: int = 0, **extra):
    """Instantiate the engine for a config row.

    ``adaptation`` in {zero, fs16, fs64, full}: everything except ``zero`` loads a
    checkpoint produced by ``tsfm4finance.bench.trainers`` for this (model, series,
    budget) triple, so the prediction stage never has to know how the weights were
    obtained.
    """
    fam, key = spec.family, spec.key
    # One resolution point: ``weights`` may be a Hub id, a cached Hub id, or a
    # directory produced by ``git clone`` / ``hf download`` under $TSFM_WEIGHTS_ROOT.
    weights = _resolve(spec.weights) or spec.weights
    from pathlib import Path as _P
    if adaptation != "zero":
        # an adapted cell must load the checkpoint that the family trainer produced;
        # never silently fall back to base weights (that would mislabel the result)
        ft_ckpt = _resolve(ft_ckpt) if ft_ckpt else ""
        from pathlib import Path as _P
        if not ft_ckpt or not _P(ft_ckpt).exists():
            raise FileNotFoundError(
                f"fine-tuned checkpoint for {spec.key} budget={adaptation} not found "
                f"({ft_ckpt or 'unset'}); train it first:  "
                f"python -m tsfm4finance.bench.trainers --model {spec.key} "
                f"--series <SERIES> --budget {adaptation}")
    else:
        ft_ckpt = None
    primary = ft_ckpt if adaptation != "zero" else weights
    _adapted = adaptation != "zero"
    _as_dir = _adapted and _P(ft_ckpt).is_dir()      # save_pretrained-style checkpoint
    primary = ft_ckpt if (_adapted and _as_dir) else weights
    kw = dict(device=device, amp=amp, batch=batch, ctx_cap=ctx_cap or spec.max_context,
              seed=seed, key=key, **extra)

    if fam == "baseline":
        from . import baselines as B
        cls = {"naive": B.NaiveEngine, "seasonal_naive": B.SeasonalNaiveEngine,
               "moving_average": B.MovingAverageEngine, "arima": B.ArimaEngine,
               "arima_auto": B.ArimaAutoEngine, "ets": B.EtsEngine,
               "theta": B.ThetaEngine}[key]
        return cls(ctx_cap=ctx_cap or spec.max_context)

    if fam == "timesfm25":
        from .timesfm import TimesFM25Engine
        return TimesFM25Engine(checkpoint=ft_ckpt if adaptation != "zero" else None, **kw)

    if fam == "timesfm30":
        from .timesfm import TimesFM30Engine
        if adaptation != "zero":
            return TimesFM30Engine(checkpoint=primary, **kw)
        return TimesFM30Engine(checkpoint=weights, **kw)

    if fam == "chronos":
        from .chronos import ChronosEngine
        return ChronosEngine(checkpoint=primary, **kw)

    if fam == "moirai":
        from .moirai import MoiraiEngine
        return MoiraiEngine(checkpoint=primary, **kw)

    if fam == "moment":
        from .moment import MomentEngine
        e = MomentEngine(checkpoint=weights, **kw)
        if adaptation != "zero" and ft_ckpt:
            e.load_finetuned(ft_ckpt)
        return e

    if fam == "ttm":
        from .granite import TTMEngine
        return TTMEngine(checkpoint=primary, **kw)

    if fam == "flowstate":
        from .granite import FlowStateEngine
        return FlowStateEngine(checkpoint=primary, **kw)

    if fam == "toto2":
        from .toto import Toto2Engine
        return Toto2Engine(checkpoint=primary, **kw)

    if fam == "tirex":
        from .tirex import TiRexEngine
        return TiRexEngine(checkpoint=spec.weights, **kw)

    if fam == "tirex2":
        from .tirex import TiRex2Engine
        return TiRex2Engine(checkpoint=spec.weights, **kw)

    if fam == "timemoe":
        from .timemoe import TimeMoEEngine
        return TimeMoEEngine(checkpoint=primary, **kw)

    if fam == "kronos":
        from .kronos import KronosEngine
        tok = _resolve(extra.pop("tokenizer", "NeoQuasar/Kronos-Tokenizer-base"))
        return KronosEngine(checkpoint=primary, tokenizer=tok, **kw)

    raise KeyError(f"no engine implemented for family {fam!r} (model {key!r})")


__all__ = ["build_engine", "LEVELS", "Pred", "to_levels", "quantiles_from_samples"]


_raw_build_engine = build_engine


def build_engine(spec, *, adaptation: str = "zero", ft_ckpt: str | None = None, **kw):
    """Public entry point: resolve weights, build from BASE weights, then apply the
    family trainer's state dict for adapted cells (unless the checkpoint is a
    save_pretrained-style directory, which is loaded directly)."""
    e = _raw_build_engine(spec, adaptation=adaptation, ft_ckpt=ft_ckpt, **kw)
    if adaptation != "zero" and ft_ckpt and not getattr(e, "ft_meta", None):
        from pathlib import Path as _P
        if str(ft_ckpt).endswith(".pt") and _P(ft_ckpt).exists():
            e.load_finetuned(str(ft_ckpt))
    return e
