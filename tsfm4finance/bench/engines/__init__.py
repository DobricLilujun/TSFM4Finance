"""Engine registry: config key -> engine instance, inside the family's own env.

``build_engine`` is the only place that knows which class implements which family,
so adding a model means adding a config row (plus, if needed, a module here).
"""
from __future__ import annotations

from .base import LEVELS, Pred, quantiles_from_samples, to_levels


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
            return TimesFM30Engine(checkpoint=ft_ckpt or spec.weights, **kw)
        return TimesFM30Engine(checkpoint=spec.weights, **kw)

    if fam == "chronos":
        from .chronos import ChronosEngine
        return ChronosEngine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "moirai":
        from .moirai import MoiraiEngine
        return MoiraiEngine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "moment":
        from .moment import MomentEngine
        e = MomentEngine(checkpoint=spec.weights, **kw)
        if adaptation != "zero" and ft_ckpt:
            e.load_finetuned(ft_ckpt)
        return e

    if fam == "ttm":
        from .granite import TTMEngine
        return TTMEngine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "flowstate":
        from .granite import FlowStateEngine
        return FlowStateEngine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "toto2":
        from .toto import Toto2Engine
        return Toto2Engine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "tirex":
        from .tirex import TiRexEngine
        return TiRexEngine(checkpoint=spec.weights, **kw)

    if fam == "tirex2":
        from .tirex import TiRex2Engine
        return TiRex2Engine(checkpoint=spec.weights, **kw)

    if fam == "timemoe":
        from .timemoe import TimeMoEEngine
        return TimeMoEEngine(checkpoint=ft_ckpt or spec.weights, **kw)

    if fam == "kronos":
        from .kronos import KronosEngine
        tok = extra.pop("tokenizer", "NeoQuasar/Kronos-Tokenizer-base")
        return KronosEngine(checkpoint=ft_ckpt or spec.weights, tokenizer=tok, **kw)

    raise KeyError(f"no engine implemented for family {fam!r} (model {key!r})")


__all__ = ["build_engine", "LEVELS", "Pred", "to_levels", "quantiles_from_samples"]
