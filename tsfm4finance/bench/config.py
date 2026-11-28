"""Declarative benchmark grid + model registry.

Everything the sweep needs is data, not code: which models exist, which python
environment can import them, what they are capable of (forecast / quantiles /
embeddings / classification head / fine-tunable), and which cells the grid
contains.  The config hash is stamped into every cell so a partial run knows
exactly which grid produced it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ModelSpec:
    key: str
    family: str
    env: str
    weights: str = ""
    params_m: float = 0.0
    max_context: int = 512
    max_horizon: int = 128
    freq: str = "D"
    license: str = ""
    needs: list[str] = field(default_factory=list)     # input columns: close/open/high/low/volume
    caps: list[str] = field(default_factory=list)      # forecast quantiles embeddings clf_head anomaly_head finetune sample_paths
    tier: str = "A"
    vram_mb: int = 1024
    ft: dict = field(default_factory=dict)             # trainer defaults
    notes: str = ""

    def can(self, cap: str) -> bool:
        return cap in self.caps


@dataclass
class Grid:
    cfg: dict
    path: Path

    # ---------------------------------------------------------------- environments
    @property
    def envs(self) -> dict[str, str]:
        return dict(self.cfg.get("envs") or {})

    def interpreter(self, env_name: str) -> str:
        """Interpreter for an env *name* (used by the dispatcher: a worker must run
        in the family's own environment, never in the driver's)."""

        class _E:
            env = env_name
        return self.python_for(_E())

    def python_for(self, m: ModelSpec) -> str:
        """Resolve an ``envs:`` entry to an interpreter.

        Accepts a path (``.venv2/bin/python``, ``envs/core/.venv/bin/python``) or a
        bare name (``core``), in which case the usual layouts are tried in order:
        the uv project env (``envs/<name>/.venv``), the ad-hoc venv
        (``.venv-<name>``) and ``.venv``.  That way the same config.yaml works on a
        machine built with ``scripts/setup_envs.sh`` and on one with the older
        hand-made venvs.
        """
        raw = self.envs.get(m.env, self.envs.get("core", sys.executable))
        raw = str(raw)
        root = Path(__file__).resolve().parents[2]
        cand: list[Path] = []
        p = Path(raw)
        if p.is_absolute() or ".venv" in str(p) or str(p).endswith("python"):
            cand.append(p)
        for name in dict.fromkeys((m.env, raw)):
            cand += [root / "envs" / name / ".venv" / "bin" / "python",
                     root / f".venv-{name}" / "bin" / "python"]
            if name in ("core", "sweep"):
                cand.append(root / ".venv2" / "bin" / "python")
            cand.append(root / ".venv" / "bin" / "python")
        for c in cand:
            if c.exists():
                return str(c)
        return raw

    def models(self, tiers: tuple[str, ...] | None = None,
               only: tuple[str, ...] | None = None) -> dict[str, ModelSpec]:
        out: dict[str, ModelSpec] = {}
        for key, d in (self.cfg.get("models") or {}).items():
            if tiers and d.get("tier", "A") not in tiers:
                continue
            if only and key not in only:
                continue
            spec = ModelSpec(key=key, family=d.get("family", key), env=d.get("env", "core"),
                             weights=d.get("weights", ""), params_m=float(d.get("params_m", 0)),
                             max_context=int(d.get("max_context", 512)),
                             max_horizon=int(d.get("max_horizon", 128)),
                             freq=d.get("freq", "D"), license=d.get("license", ""),
                             needs=list(d.get("needs") or []), caps=list(d.get("caps") or []),
                             tier=d.get("tier", "A"), vram_mb=int(d.get("vram_mb", 1024)),
                             ft=dict(d.get("ft") or {}), notes=d.get("notes", ""))
            out[key] = spec
        return out

    def series_groups(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in (self.cfg.get("series_groups") or {}).items()}

    def series(self, groups: tuple[str, ...] | None = None,
               only: tuple[str, ...] | None = None,
               exclude: tuple[str, ...] = ()) -> list[str]:
        g = self.series_groups()
        names: list[str] = []
        for k in (groups or g.keys()):
            names += g.get(k, [])
        if only:
            names = [n for n in names if n in only]
        return [n for n in names if n not in exclude]

    def section(self, name: str, default=None):
        return dict(self.cfg.get(name) or (default or {}))

    # ---------------------------------------------------------------- hashing
    def digest(self) -> str:
        blob = json.dumps(self.cfg, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Grid({self.path.name}, hash={self.digest()}, models={len(self.models())})"


def load_grid(path: str | Path | None = None) -> Grid:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"benchmark config not found: {p}")
    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read bench config.yaml "
                               "(pip install pyyaml)")
        cfg = yaml.safe_load(p.read_text())
    else:
        cfg = json.loads(p.read_text())
    return Grid(cfg=cfg, path=p)
