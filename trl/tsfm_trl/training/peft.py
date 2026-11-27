"""Which parameters move, and how we can prove it afterwards.

Three regimes, all recorded in the checkpoint (a head-only run must never be
reportable as a full fine-tune):

* ``full``       — every parameter trainable;
* ``heads_only`` — backbone frozen, only the prediction head(s) trainable;
* ``lora``       — backbone frozen, low-rank adapters on the projections; the
  adapters are **merged into the base weights before saving**, so a checkpoint is
  loadable by the plain inference path (no custom module needed to use it).
"""
from __future__ import annotations

from typing import Iterable

from .config import DEFAULT_HEAD_PREFIXES


def count_trainable(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _match(name: str, prefix: str) -> bool:
    return name.startswith(prefix) or ("." + prefix) in name


def set_trainable(module, mode: str = "full", head_prefixes: Iterable[str] = ()):
    """Apply ``full`` / ``heads_only`` and return the list of parameter names (or
    patterns) that are trainable — this list is what the checkpoint stores."""
    import torch
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"expected nn.Module, got {type(module).__name__}")
    prefixes = tuple(head_prefixes) or DEFAULT_HEAD_PREFIXES
    if mode == "full":
        module.requires_grad_(True)
        return ["*all*"]
    module.requires_grad_(False)
    touched = []
    for name, p in module.named_parameters():
        # a prefix matches at the start *or* at any dot-segment boundary, so an
        # adapter can write head_prefixes=("linear_head",) and hit
        # "model.linear_head.weight" in a nested vendor module
        if any(_match(name, h) for h in prefixes):
            p.requires_grad_(True)
            touched.append(name)
    if not touched:                       # nothing matched: fail loudly rather than
        raise ValueError(                 # quietly training zero parameters
            f"peft='heads_only' matched no parameters; head_prefixes={prefixes}; "
            f"declare them on the adapter (head_prefixes=...)")
    return touched


# ------------------------------------------------------------------------ LoRA
#: the broad net used when a family names its layers something unusual
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
                        "fc1", "fc2", "proj", "lin", "linear", "dense",
                        "attn", "mix", "up", "down",
                        # bare leaves: chronos (bolt + t5) and several T5-derived families
                        # name their attention projections q/k/v/o
                        "q", "k", "v", "o", "wi", "wo")


def linear_leaves(module, limit: int = 12) -> list[str]:
    """Leaf names of the ``nn.Linear``s in a module — the diagnostics that make a
    target-list mismatch fixable instead of a guessing game."""
    import torch
    out = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            out.append(name.split(".")[-1])
            if len(out) >= limit:
                break
    return sorted(set(out))


def _inject(module, tg, r, alpha):
    import torch
    import torch.nn.functional as F
    injected = []
    for name, m in module.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if not (leaf in tg or any(leaf.endswith(t) for t in tg)):
            continue
        scale = (alpha or 2 * r) / r
        m.lora_A = torch.nn.Parameter(torch.randn(r, m.in_features).to(m.weight) * (1.0 / r))
        m.lora_B = torch.nn.Parameter(torch.zeros(m.out_features, r).to(m.weight))
        m._lora_scale = scale
        orig = m.forward

        def fwd(x, _m=m, _orig=orig):
            return _orig(x) + F.linear(F.linear(x, _m.lora_A), _m.lora_B) * _m._lora_scale

        m.forward = fwd
        m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)
        injected.append(name or leaf)
    return injected


def apply_lora(module, r: int = 8, alpha: float | None = None,
               targets: Iterable[str] = DEFAULT_LORA_TARGETS):
    """Inject LoRA into every matching ``nn.Linear``. Returns the injected names.

    A target list that matches nothing is usually a naming difference between families
    (Chronos-Bolt has ``q``/``k``/``v``, uni2ts has ``wqkv``, MOMENT has ``q``), so the
    configured list is tried first, then the broad default, and only then is it an
    error — and the error names the Linear layers that *do* exist.
    """
    tg = tuple(targets or ())
    injected = _inject(module, tg, r, alpha) if tg else []
    if not injected and tg and tg != DEFAULT_LORA_TARGETS:
        injected = _inject(module, DEFAULT_LORA_TARGETS, r, alpha)
    if not injected:
        raise ValueError(f"peft='lora' matched no nn.Linear; targets={tg}. "
                         f"Linear layers present: {linear_leaves(module)}")
    return injected


def make_trainable(model, module, peft: str = "heads_only", *, r: int = 8,
                   targets: Iterable[str] | None = None):
    """The one entry point both trainers use, so ``peft='lora'`` cannot mean
    'inject adapters' in one trainer and 'train the head' in another."""
    if peft == "lora":
        # Freeze the trunk *first*. ``_inject`` freezes the Linear layers it wraps and
        # enables their adapters, but it never touches everything else — and every HF
        # checkpoint loads with requires_grad=True, so without this pass "lora" quietly
        # meant "full fine-tune, plus a few adapters": t0-gamma reported 156 M trainable
        # parameters, which is its trunk, not an adapter budget.
        for _name, prm in module.named_parameters():
            prm.requires_grad_(False)
        tg = tuple(targets or ()) + tuple(getattr(model, "lora_targets", ()) or ())
        trained = apply_lora(module, r=r, targets=tg or DEFAULT_LORA_TARGETS)
        if getattr(model, "lora_trains_head", False):
            # a head the adapter *re-initialises* (Sundial rebuilds its TimeFlow head at
            # the training horizon) has no adapter and no pretrained weights to nudge:
            # it has to be trained outright next to the trunk adapters
            prefixes = tuple(getattr(model, "head_prefixes", ()) or ()) or DEFAULT_HEAD_PREFIXES
            for name, prm in module.named_parameters():
                if any(_match(name, h) for h in prefixes):
                    prm.requires_grad_(True)
                    trained.append(name)
        return trained
    return set_trainable(module, peft, getattr(model, "head_prefixes", ()))


def merge_lora(module) -> int:
    """Fold the adapters into the base weights so the state dict is plain."""
    import torch
    n = 0
    for name, m in list(module.named_modules()):
        if getattr(m, "lora_B", None) is None:
            continue
        with torch.no_grad():
            m.weight += (m.lora_B @ m.lora_A) * m._lora_scale
        del m.lora_A
        del m.lora_B
        del m._lora_scale
        try:
            del m.forward                     # instance attr -> class forward again
        except Exception:                     # noqa: BLE001
            pass
        n += 1
    return n


def move_deep(module, device: str) -> int:
    """Move a model *and* its non-registered tensor attributes to ``device``.

    Several vendor TSFMs cache scratch tensors as plain attributes (FlowState keeps
    patch/quantile tables outside the parameter/buffer graph). ``module.to()`` leaves
    those behind, and the next forward fails with a device mismatch that looks like an
    adapter bug. Returns how many stray tensors were moved, so the matrix can say so.
    """
    import torch
    module.to(device)
    tgt = torch.device(device)
    n = 0
    for m in module.modules():
        for name, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device != tgt:
                try:
                    setattr(m, name, v.to(tgt))
                    n += 1
                except Exception:                                      # noqa: BLE001
                    pass
    return n


def optimiser(module, lr: float, weight_decay: float = 0.01):
    import torch
    params = [p for p in module.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def maybe_scheduler(opt, warmup: int, total_steps: int):
    if warmup <= 0:
        return None
    import torch

    def factor(step):
        return min(1.0, (step + 1) / max(1, warmup))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)
