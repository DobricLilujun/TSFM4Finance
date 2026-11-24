"""Public model API: :class:`ModelHub`.

    from tsfm_trl import ModelHub
    m = ModelHub.load("chronos-2")                       # registry key
    m = ModelHub.load("hf://mistralai/Timestral-28M")    # or a bare HF id
    m = ModelHub.load("./models/amazon/chronos-2")       # or a local directory
    p = m.forecast(context, horizon=21)

``load`` never raises a bare ImportError: an unloaded model reports *why* (missing
package, missing weights, bad toolchain) so the compatibility matrix stays honest.
"""
from __future__ import annotations

import importlib.util

from .base import (CAPS, LEVELS, MAX_CONTEXT, Pred, TSFMModel, autocast,
                   levels_from_count, quantiles_from_samples, to_levels)
from .registry import ModelSpec, Registry, default_registry
from ..errors import TSFMTRLError, UnknownModelError, UnavailableError

__all__ = ["ModelHub", "Registry", "ModelSpec", "Pred", "TSFMModel", "LEVELS",
           "MAX_CONTEXT", "to_levels", "levels_from_count", "quantiles_from_samples",
           "autocast", "CAPS"]


class ModelHub:
    """Factory + inventory. Stateless: the same call returns an equivalent model."""

    @staticmethod
    def registry() -> Registry:
        return default_registry()

    @staticmethod
    def list(tier: str | None = None, cap: str | None = None, family: str | None = None,
             env: str | None = None, task: str | None = None) -> list[ModelSpec]:
        """``task`` is one of forecast | classification | reward | anomaly (or a raw cap)."""
        return ModelHub.registry().models(tier=tier, cap=cap, family=family, env=env, task=task)

    @staticmethod
    def lookup(name: str) -> ModelSpec:
        """Resolve a registry key, an HF id, or a local weights directory name."""
        spec = ModelHub.registry().match(name)
        if spec is None:
            from ..errors import UnknownModelError
            raise UnknownModelError(name, ModelHub.registry().keys())
        return spec

    @staticmethod
    def info(key: str) -> ModelSpec:
        return ModelHub.registry().get(key)

    # ------------------------------------------------------------------ import probe
    @staticmethod
    def importable_modules() -> dict[str, bool]:
        """Which model-family packages this interpreter can import."""
        probe = {
            "timesfm": "timesfm", "chronos": "chronos", "uni2ts": "uni2ts",
            "momentfm": "momentfm", "tsfm_public": "tsfm_public", "toto2": "toto2",
            "tirex": "tirex", "tirex2": "tirex2", "transformers": "transformers",
            "kronos": "model.kronos", "timestral": "timestral", "hirondel": "hirondel",
            "t0": "t0", "yug": "yug", "visionts": "visionts",
            "statsmodels": "statsmodels", "sklearn": "sklearn", "torch": "torch",
        }
        out = {}
        for name, mod in probe.items():
            try:
                out[name] = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError, ModuleNotFoundError):
                out[name] = False
        return out

    # ------------------------------------------------------------------ load
    @staticmethod
    def load(name: str, *, adaptation: str = "zero", checkpoint: str | None = None,
             device: str = "auto", amp: str = "fp32", batch: int = 16,
             ctx_cap: int | None = None, seed: int = 0, **extra) -> TSFMModel:
        """Instantiate a model.

        ``adaptation``: ``"zero"`` (base weights) or a budget name with
        ``checkpoint=`` pointing at a checkpoint written by :mod:`tsfm_trl.training`.
        An adapted cell that cannot load its checkpoint raises — it never quietly
        falls back to base weights.
        """
        from ..weights import resolve as resolve_weights
        reg = ModelHub.registry()
        spec = reg.match(name)
        if spec is None and name.startswith(("hf://", "hf:")):
            spec = _spec_from_hf(name.split("://", 1)[1] if "://" in name else name.split(":", 1)[1])
        if spec is None:
            spec = _spec_from_path(name)
        if spec is None:
            raise UnknownModelError(name, reg.keys())

        device = _resolve_device(device)
        cls = reg.adapter_class(spec)
        weights = resolve_weights(spec.weights) if spec.weights else None
        kwargs = dict(spec.extra)
        kwargs.update(extra)
        if weights:
            kwargs["checkpoint"] = weights
        kwargs.setdefault("source_id", spec.weights)     # HF id survives resolution
        try:
            model = cls(device=device, amp=amp, batch=batch, ctx_cap=ctx_cap or spec.max_context,
                        seed=seed, **kwargs)
        except (ImportError, ModuleNotFoundError) as e:
            raise UnavailableError(spec.key, f"missing package: {e}", spec.install) from e
        except Exception as e:                                    # noqa: BLE001
            raise UnavailableError(spec.key, f"{type(e).__name__}: {e}", spec.install) from e

        model.spec = spec
        if adaptation and adaptation != "zero":
            if not checkpoint:
                from pathlib import Path
                cand = Path("reports/ft") / f"{spec.key}__{adaptation}.pt"
                checkpoint = str(cand) if cand.exists() else None
            if not checkpoint:
                raise TSFMTRLError(
                    f"{spec.key}: no fine-tuned checkpoint for adaptation={adaptation!r}; "
                    f"train it first (tsfm-trl fit --model {spec.key})")
            model.load_finetuned(checkpoint)
            model.adaptation = adaptation
        else:
            model.adaptation = "zero"
        return model


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                             # noqa: BLE001
        return "cpu"


# heuristics so a bare HF id still lands on the right adapter
_HF_HINTS = (
    ("timesfm-3", "timesfm30", "timesfm:TimesFM30"),
    ("timesfm-2.5", "timesfm25", "timesfm:TimesFM25"),
    ("timesfm-2.0", "timesfm20", "timesfm:TimesFM20"),
    ("timesfm", "timesfm25", "timesfm:TimesFM25"),
    ("chronos-2", "chronos2", "chronos:Chronos2"),
    ("chronos-bolt", "chronos_bolt", "chronos:ChronosBolt"),
    ("chronos-t5", "chronos_t5", "chronos:ChronosT5"),
    ("moirai-2.0", "moirai2", "moirai:Moirai2"),
    ("moirai-moe", "moirai_moe", "moirai:MoiraiMoE"),
    ("moirai", "moirai", "moirai:Moirai"),
    ("moment", "moment", "moment:Moment"),
    ("flowstate", "flowstate", "granite:FlowState"),
    ("ttm", "ttm", "granite:TTM"),
    ("patchtst-fm", "patchtst", "granite:PatchTSTFM"),
    ("toto-2", "toto2", "toto:Toto2"),
    ("toto", "toto1", "toto:Toto1"),
    ("tirex-2", "tirex2", "tirex:TiRex2"),
    ("tirex", "tirex", "tirex:TiRex"),
    ("timemoe", "timemoe", "timemoe:TimeMoE"),
    ("sundial", "sundial", "sundial:Sundial"),
    ("kronos", "kronos", "kronos:Kronos"),
    ("kairos", "kairos", "kairos:Kairos"),
    ("timestral", "timestral", "timestral:Timestral"),
    ("hirondel", "hirondel", "hirondel:Hirondel"),
    ("t0", "t0", "t0:T0"),
    ("yug", "yug", "yug:Yug"),
    ("aurora", "aurora", "aurora:Aurora"),
    ("visionts", "visionts", "visionts:VisionTSpp"),
)


def _spec_from_hf(hf_id: str) -> ModelSpec | None:
    low = hf_id.lower()
    for hint, family, adapter in _HF_HINTS:
        if hint in low:
            return ModelSpec(key=hf_id.split("/")[-1], family=family, adapter=adapter,
                             weights=hf_id, caps=("forecast",), loss="none", tier="B",
                             notes="inferred from a bare Hugging Face id")
    return None


def _spec_from_path(path: str) -> ModelSpec | None:
    from pathlib import Path
    p = Path(path).expanduser()
    if not p.exists():
        return None
    spec = _spec_from_hf(p.name)
    if spec:
        spec.weights = str(p)
    return spec
