"""Full fine-tuning of Google TimesFM 2.5-200M (decision Q3: full FT, few-shot ladder).

The pip package ships inference only, so this module trains the underlying
`TimesFM_2p5_200M_torch_module` directly, reproducing the model's own normalization
pipeline so that training and inference agree:

  inference (`compile()` + `forecast()`, normalize_inputs=True):
    global REVIN over the context  ->  patchify (p=32)  ->  running-stat REVIN per patch
    -> transformer -> per-patch renormalize -> global REVIN reverse.
  training here:
    same two normalizations, supervising the LAST patch's 128-step output patch
    (o = output_patch_len = 128) in global-normalized units.

Loss = pinball over the 9 quantile channels (levels 0.1..0.9 = channels 1..9)
     + L1 on the point channel (index 5 = median), matching how the checkpoint decodes.

Data discipline: training windows are drawn ONLY from the dataset's train split.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CONTEXT = 256          # fine-tune context (business days); matched zero-shot runs use the same
TARGET_LEN = 128       # the model's output patch length (o)
STRIDE = 4
QUANT_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
POINT_CHANNEL = 5
EPOCHS_BY_K = {16: 10, 64: 6, 256: 4, None: 3}     # None = full train split


@dataclass
class FTResult:
    checkpoint: str
    dataset: str
    k_windows: int
    context: int
    steps: int
    epochs: int
    lr: float
    batch_size: int
    final_loss: float
    first_loss: float
    seconds: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def make_windows(series: np.ndarray, context: int = CONTEXT, target_len: int = TARGET_LEN,
                 stride: int = STRIDE, max_windows: int | None = None,
                 seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sliding (context, next-target_len) pairs from a single series."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    L = context + target_len
    if len(s) < L:
        return []
    starts = list(range(0, len(s) - L + 1, max(1, stride)))
    wins = [(s[i:i + context], s[i + context:i + L]) for i in starts]
    if max_windows is not None and len(wins) > max_windows:
        rng = np.random.RandomState(seed)
        wins = [wins[i] for i in rng.choice(len(wins), size=max_windows, replace=False)]
    return wins


def _patchify_and_normalize(module, x: np.ndarray, y: np.ndarray, device,
                            global_norm: bool = True):
    """Reproduce compile(normalize_inputs=True) + decode() prefill normalization.

    Returns (normed_patches, masks, patch_mu, patch_sigma, y_global_norm) all as tensors,
    where y_global_norm is the target expressed in the same globally-normalized units the
    decoder output is renormalized back into.
    """
    import torch
    from timesfm.torch import util as tfutil

    p = module.p
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)      # (1, C)
    mu0 = xt.mean(dim=-1, keepdim=True)
    sd0 = xt.std(dim=-1, keepdim=True).clamp_min(1e-6)
    if global_norm:
        xin = (xt - mu0) / sd0
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        yin = (yt - mu0.squeeze(-1)) / sd0.squeeze(-1)
    else:
        xin, yt = xt, torch.tensor(y, dtype=torch.float32, device=device)
        yin = yt

    patched = torch.reshape(xin, (1, -1, p))
    masks = torch.zeros_like(patched, dtype=torch.bool)
    n = torch.zeros(1, device=device)
    mu = torch.zeros(1, device=device)
    sg = torch.zeros(1, device=device)
    mus, sgs = [], []
    for i in range(patched.shape[1]):
        (n, mu, sg), _ = tfutil.update_running_stats(n, mu, sg, patched[:, i], masks[:, i])
        mus.append(mu)
        sgs.append(sg)
    cmu = torch.stack(mus, dim=1)          # (1, P)
    csd = torch.stack(sgs, dim=1)
    normed = tfutil.revin(patched, cmu, csd, reverse=False)
    normed = torch.where(masks, 0.0, normed)
    return normed, masks, cmu, csd, yin


def _loss(module, normed, masks, cmu, csd, yin, amp: str | None = None):
    import torch
    from timesfm.torch import util as tfutil

    if amp:
        dt = torch.bfloat16 if amp == "bf16" else torch.float16
        with torch.autocast(device_type=normed.device.type, dtype=dt):
            (_, _, normed_outputs, _), _ = module(normed, masks, None)
        normed_outputs = normed_outputs.float()          # loss math stays in fp32
    else:
        (_, _, normed_outputs, _), _ = module(normed, masks, None)
    last = normed_outputs[:, -1, :]                                  # (1, o*q)
    o, q = module.o, module.q
    last = torch.reshape(last, (1, o, q))
    renorm = tfutil.revin(last, cmu[:, -1:], csd[:, -1:], reverse=True)   # -> global units
    renorm = renorm[:, :yin.shape[-1], :]                            # (1, H, q)
    ch = renorm[..., QUANT_CHANNELS]                                  # (1, H, 9)
    y = yin.unsqueeze(-1)                                             # (1, H, 1)
    r = y - ch
    lev = torch.tensor(LEVELS, dtype=r.dtype, device=r.device)
    pinball = torch.maximum(lev * r, (lev - 1.0) * r).mean()
    point = torch.abs(y.squeeze(-1) - renorm[..., POINT_CHANNEL]).mean()
    return pinball + point, float(pinball.detach()), float(point.detach())


def finetune(train_close: np.ndarray, out_dir: str | Path, dataset: str = "",
             k_windows: int | None = 64, context: int = CONTEXT,
             target_len: int = TARGET_LEN, stride: int = STRIDE,
             lr: float = 1e-5, batch_size: int = 8, epochs: int | None = None,
             seed: int = 0, global_norm: bool = True, verbose: bool = False,
             device: str | None = None, amp: str | None = None) -> FTResult:
    """Full (all-parameter) fine-tune of the 200M TimesFM module; saves a loadable ckpt.

    Memory knobs for small GPUs: ``batch_size``, ``context`` (shorter windows) and
    ``amp="bf16"|"fp16"`` (activations in half precision, loss computed in fp32).
    """
    import torch
    import timesfm

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    wins = make_windows(train_close, context, target_len, stride,
                        max_windows=k_windows, seed=seed)
    if not wins:
        raise ValueError(f"not enough rows to build fine-tune windows "
                         f"(have {len(train_close)}, need {context + target_len})")
    ep = int(epochs if epochs is not None else EPOCHS_BY_K.get(k_windows, 3))

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID, torch_compile=False)
    module = wrapper.model
    if device:
        module.to(device)
    device = next(module.parameters()).device
    module.train()
    for prm in module.parameters():
        prm.requires_grad_(True)

    optim = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)
    bs = batch_size
    losses: list[float] = []
    steps = 0
    first = last = float("nan")
    try:
        for e in range(ep):
            rng = np.random.RandomState(seed + e)
            order = rng.permutation(len(wins))
            for i0 in range(0, len(order), bs):
                sel = order[i0:i0 + bs]
                accum = []
                optim.zero_grad(set_to_none=True)
                for j in sel:
                    ctx, tgt = wins[int(j)]
                    try:
                        normed, masks, cmu, csd, yin = _patchify_and_normalize(
                            module, ctx, tgt, device, global_norm)
                        l, _, _ = _loss(module, normed, masks, cmu, csd, yin, amp=amp)
                        accum.append(l / len(sel))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if len(sel) > 1:
                            continue
                        raise
                if not accum:
                    raise RuntimeError("could not form a training batch (OOM)")
                total = torch.stack(accum).sum()
                total.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optim.step()
                lv = float(total.detach())
                losses.append(lv)
                if not np.isfinite(first):
                    first = lv
                last = lv
                steps += 1
                if verbose and steps % 20 == 0:
                    print(f"    step {steps:4d} loss {lv:.4f}", flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs > 1:
            optim.close() if hasattr(optim, "close") else None
            del optim
            return finetune(train_close, out, dataset, k_windows, context, target_len,
                            stride, lr, batch_size=max(1, bs // 2), epochs=epochs, seed=seed,
                            global_norm=global_norm, verbose=verbose,
                            device=str(device) if device else None, amp=amp)
        raise

    wrapper.save_pretrained(str(out))
    meta = {"dataset": dataset, "k_windows": k_windows, "n_windows_used": len(wins),
            "trainable_parameters": int(sum(p.numel() for p in module.parameters()
                                            if p.requires_grad)),
            "amp": amp or "fp32", "device": str(device),
            "context": context, "target_len": target_len, "epochs": ep, "steps": steps,
            "lr": lr, "batch_size": bs, "seed": seed, "loss_first": first, "loss_last": last,
            "loss_curve": losses[:: max(1, len(losses) // 40)]}
    (out / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
    res = FTResult(checkpoint=str(out), dataset=dataset, k_windows=int(k_windows or len(wins)),
                   context=context, steps=steps, epochs=ep, lr=lr, batch_size=bs,
                   first_loss=first, final_loss=last, seconds=time.time() - t0, seed=seed)
    return res
