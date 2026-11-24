"""Model registry: one YAML row per (family, checkpoint), one import point.

``ModelHub.load("chronos-2")`` is the only thing a user needs to type; everything
else (which class implements the family, which weights, which capabilities, which
python environment) is declared in :data:`tsfm_trl/models.yaml`.
"""
from __future__ import annotations

from pathlib import Path

import importlib
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache

import yaml

from .base import CAPS
from ..errors import UnknownModelError

# models.yaml is package data at the package root, not next to this module
YAML_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


@dataclass
class ModelSpec:
    key: str
    family: str
    adapter: str                       # "module:Class" inside tsfm_trl.model.adapters
    weights: str | None = None
    caps: tuple[str, ...] = ("forecast",)
    loss: str = "none"                 # quantile | token | vendor | none
    env: str = "core"
    tier: str = "B"
    license: str = ""
    max_context: int = 512
    max_horizon: int = 512
    needs: tuple[str, ...] = ("close",)
    install: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def can(self, cap: str) -> bool:
        return cap in set(self.caps)

    @property
    def trainable(self) -> bool:
        return self.loss in ("quantile", "token", "vendor") or self.can("finetune")

    def to_dict(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        return d


class Registry:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else YAML_PATH
        raw = yaml.safe_load(self.path.read_text())
        self.envs: dict[str, str] = dict(raw.get("envs") or {})
        self._models: dict[str, ModelSpec] = {}
        for key, row in (raw.get("models") or {}).items():
            row = dict(row)
            spec = ModelSpec(
                key=key,
                family=row.pop("family"),
                adapter=row.pop("adapter"),
                weights=row.pop("weights", None),
                caps=tuple(row.pop("caps", ("forecast",))),
                loss=str(row.pop("loss", "none")),
                env=str(row.pop("env", "core")),
                tier=str(row.pop("tier", "B")),
                license=str(row.pop("license", "")),
                max_context=int(row.pop("max_context", 512)),
                max_horizon=int(row.pop("max_horizon", 512)),
                needs=tuple(row.pop("needs", ("close",))),
                install=str(row.pop("install", "")),
                notes=str(row.pop("notes", "")),
                extra=row)
            bad = [c for c in spec.caps if c not in CAPS]
            if bad:
                raise ValueError(f"{key}: unknown capabilities {bad}; valid: {CAPS}")
            self._models[key] = spec

    # ------------------------------------------------------------------ lookup
    _TASK_CAPS = {"forecast": ("forecast",), "classification": ("clf_head", "embed"),
                  "reward": ("finetune",), "anomaly": ("anomaly_head", "quantiles")}

    def models(self, *, tier: str | None = None, family: str | None = None,
               cap: str | None = None, env: str | None = None,
               task: str | None = None) -> list[ModelSpec]:
        out = list(self._models.values())
        if tier:
            out = [m for m in out if m.tier in tier.split(",")]
        if family:
            pats = family.split(",")
            out = [m for m in out if any(fnmatch(m.family, p) or m.family in pats for p in pats)]
        if cap:
            out = [m for m in out if m.can(cap)]
        if env:
            pats = env.split(",")
            out = [m for m in out if any(fnmatch(m.env, p) or m.env in pats for p in pats)]
        if task:
            want = self._TASK_CAPS.get(task, (task,))
            out = [m for m in out if any(m.can(c) for c in want)]
        return sorted(out, key=lambda m: (m.tier, m.family, m.key))

    def keys(self) -> list[str]:
        return sorted(self._models)

    def get(self, key: str) -> ModelSpec:
        try:
            return self._models[key]
        except KeyError:
            raise UnknownModelError(key, sorted(self._models)) from None

    def match(self, name: str) -> ModelSpec | None:
        """Best-effort resolution of a free-form name: registry key, HF id, or a
        local path whose directory name looks like a known checkpoint."""
        if name in self._models:
            return self._models[name]
        low = name.lower().rstrip("/")
        base = low.split("/")[-1]
        for m in self._models.values():
            if m.weights and m.weights.lower() == low:
                return m
        for m in self._models.values():
            if m.weights and m.weights.lower().split("/")[-1] == base:
                return m
        return None

    # ------------------------------------------------------------------ classes
    def adapter_class(self, spec: ModelSpec):
        mod_name, cls_name = spec.adapter.split(":")
        mod = importlib.import_module(f"tsfm_trl.model.adapters.{mod_name}")
        try:
            return getattr(mod, cls_name)
        except AttributeError:
            raise UnknownModelError(spec.key,
                                    f"adapter {spec.adapter} not found in "
                                    f"tsfm_trl/model/adapters/{mod_name}.py") from None


@lru_cache(maxsize=8)
def default_registry() -> Registry:
    return Registry()
