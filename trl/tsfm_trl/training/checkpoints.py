"""Checkpoint format.

    {"model": <state_dict>, "meta": {...}, "config": {...}, "loss": [[step, loss], ...]}

``meta`` always carries: model key, dataset name, budget, ``split_end`` (the last
timestamp the windows could come from), trainable-parameter count, the trained
parameter names, the device/amp/ctx/batch profile that *actually* ran, and whether
it was a smoke run. That is what lets a report say "fs64 full-parameters fine-tune,
40→120 steps, 120 windows, train span ended 2026-05-31" without trusting memory.
"""
from __future__ import annotations

import json
from pathlib import Path


def save_ckpt(path: str | Path, module, meta: dict, config: dict | None = None,
              loss_curve: list | None = None, extra: dict | None = None) -> str:
    import torch
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": module.state_dict() if hasattr(module, "state_dict") else module,
             "meta": meta, "config": config or {}, "loss": loss_curve or []}
    if extra:
        state.update(extra)
    torch.save(state, p)
    side = {"path": str(p), "meta": meta, "config": config or {},
            "loss_first_last": [loss_curve[0], loss_curve[-1]] if loss_curve else []}
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str))
    return str(p)


def load_ckpt(path: str | Path, map_location: str = "cpu"):
    import torch
    return torch.load(str(path), map_location=map_location, weights_only=False)


def describe(path: str | Path) -> dict:
    """Read only the metadata of a checkpoint (no weights) — used by the CLI's
    ``ls`` and by the matrix runner to check provenance."""
    obj = load_ckpt(path)
    if not isinstance(obj, dict):
        return {"path": str(path), "format": "bare state_dict"}
    n = sum(t.numel() for t in obj.get("model", {}).values() if hasattr(t, "numel"))
    return {"path": str(path), "meta": obj.get("meta", {}), "config": obj.get("config", {}),
            "tensors": n, "loss_first_last": [obj["loss"][0], obj["loss"][-1]]
            if obj.get("loss") else []}


def load_finetuned(model, path: str | Path, *, strict: bool = True, map_location: str = "cpu"):
    """Load a fine-tuned checkpoint into an already-constructed adapter.

    The point of going through this helper instead of ``load_state_dict`` is the
    audit trail: the checkpoint's ``meta`` (base id, split_end, budget, trainable
    names) is attached to the model as ``model.provenance`` and its base id is
    checked against the adapter you loaded. A mismatch is almost always "I evaluated
    the wrong checkpoint", so it raises rather than reporting a fine-tune that never
    happened.
    """
    obj = load_ckpt(path, map_location=map_location)
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    base = meta.get("base") or meta.get("model")
    if base and getattr(model, "key", None) and base != model.key and base != getattr(model, "checkpoint", None):
        raise ValueError(f"checkpoint {path} was trained from {base!r} but you loaded {model.key!r}")
    module = model.trainable() if hasattr(model, "trainable") else None
    if module is None:
        module = getattr(model, "model", None) or getattr(model, "module", None)
    if module is None or not hasattr(module, "load_state_dict"):
        raise ValueError(f"{type(model).__name__} exposes no module to load weights into")
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} "
                         f"unexpected (first: {(missing or unexpected)[:4]})")
    # restore *which* parameters are trainable. The checkpoint stores the peft mode, not
    # the full name list (that is truncated for wide heads), and re-deriving it from the
    # mode is what make_trainable does at train time — so call the same function rather
    # than guessing from a partial list. (Iterating ``getattr(m, "parameters", [])`` — a
    # method, un-called — used to raise "'method' object is not iterable".)
    from .peft import set_trainable
    restored = set_trainable(module, str(meta.get("peft") or "full"),
                             getattr(model, "head_prefixes", ()) or ())
    if not restored:
        module.requires_grad_(bool(meta.get("train", False)))
    model.provenance = dict(meta, checkpoint=str(path))
    return model
