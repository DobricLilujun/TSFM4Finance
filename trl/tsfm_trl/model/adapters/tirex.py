"""NXAI TiRex (xLSTM; v1 under a community/non-commercial licence) and TiRex-2
(Apache). Both emit a dense quantile grid, interpolated onto the shared levels.

Two real-world caveats encoded here, not discovered at run time:

* TiRex produces **one output patch (32 steps) per pass**, so longer horizons come
  from rolling the median back into the context — the rolled quantiles therefore
  understate uncertainty growth, and ``Pred.meta`` says so;
* TiRex-2's default mLSTM kernels are Triton and do not compile against some
  CUDA/Triton pairs (observed on CUDA 13.2). We probe once and fall back to the
  pure-PyTorch kernels, which are slower but numerically equivalent — the row then
  reports ``kernels=native`` instead of disappearing from the table.
"""
from __future__ import annotations

import numpy as np

from ..base import Pred, TSFMModel, to_levels


def _candidate_ids(checkpoint: str) -> list[str]:
    """HF id first, then whatever the weights resolver found (offline cache / dir)."""
    out = [checkpoint]
    try:
        from ...weights import resolve
        r = resolve(checkpoint)
        if r and r != checkpoint:
            out.append(r)
    except Exception:                                                # noqa: BLE001
        pass
    return out


class TiRex(TSFMModel):
    key, family = "tirex", "tirex"
    caps = frozenset({"forecast", "quantiles"})
    loss_kind = None
    max_context = 2048

    def __init__(self, checkpoint: str = "NX-AI/TiRex", **kw):
        super().__init__(**kw)
        import tirex
        self.checkpoint = checkpoint
        src = [self.source_id or checkpoint, checkpoint, self.source_id]
        # TiRex's loader takes a model *id* (it downloads and validates the bundle);
        # a resolved local directory is rejected as "invalid model id", so try the id
        # first and the local path second, and keep the reason from both attempts.
        errs, seen = [], set()
        for cand in [c for c in src if c and not (c in seen or seen.add(c))] + \
                self._candidates(checkpoint):
            try:
                self.model = tirex.load_model(cand, device=self.device)
                self.source = cand
                break
            except Exception as e:                                   # noqa: BLE001
                errs.append(f"{cand}: {type(e).__name__}: {e}")
        if getattr(self, "model", None) is None:
            from ...errors import UnavailableError
            raise UnavailableError(self.key, "could not load TiRex weights -> "
                                             + " | ".join(errs)[:220])
        self.max_context = int(getattr(self.model, "max_context_length", 2048))
        self.levels = self._levels()

    @staticmethod
    def _candidates(checkpoint: str) -> list[str]:
        return _candidate_ids(checkpoint)

    def _levels(self):
        for holder in (getattr(self.model, "config", None), self.model,
                       getattr(self.model, "model", None),
                       getattr(getattr(self.model, "model", None), "config", None),
                       getattr(getattr(self.model, "model", None), "checkpoint", None)):
            for attr in ("quantiles", "quantile_levels"):
                v = getattr(holder, attr, None) if holder is not None else None
                if v is not None:
                    try:
                        out = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else list(v))]
                        if len(out) >= 3:
                            return out
                    except Exception:                          # noqa: BLE001
                        pass
        return list(np.linspace(0.001, 0.999, 1000))

    def _raw(self, cs, horizon):
        out = self.model.forecast(cs, output_type="numpy", batch_size=self.batch)
        if isinstance(out, tuple):                             # v1: (quantiles, means)
            q, mean = out
            q = np.asarray(q, float)
            return (q[:, :, None], np.asarray(mean, float)) if q.ndim == 2 else (q, np.asarray(mean, float))
        q = np.asarray(out, float)
        if q.ndim == 2:
            return q[:, :, None], q
        return q, None

    def _median_index(self, lv):
        return int(np.argmin(np.abs(np.asarray(lv) - 0.5)))

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        """Roll the vendor's single output block forward to the requested horizon.

        The median column is fed back as the point path — the position in the returned
        quantile axis is the median, not `levels`, because the vendor's grid can be
        wider than the levels we report. Per-item paths live in a list of 1-D arrays:
        slicing the batch axis out of `q0` (as an earlier version did) mixed a (B, H)
        array into a 1-D context and raised for every horizon past one block, which is
        why a compatibility pass at h <= 64 never reached this code.
        """
        cs = self.trim(contexts)
        q0, _ = self._raw(cs, horizon)
        step = int(q0.shape[1])
        qs = [q0[:, : min(step, horizon), :]]
        med0 = q0.shape[-1] // 2                      # positional median of the vendor grid
        paths = [q0[i, :step, med0].copy() for i in range(q0.shape[0])]
        while step < horizon:
            grown = [np.concatenate([c, p[:step]])[-self.max_context:]
                     for c, p in zip(cs, paths)]
            q, _ = self._raw(grown, horizon)
            take = min(int(q.shape[1]), horizon - step)
            med = q.shape[-1] // 2
            qs.append(q[:, :take, :])
            for i, pth in enumerate(paths):
                paths[i] = np.concatenate([pth, q[i, :take, med]])
            step += take
        full = np.concatenate(qs, axis=1)[:, :horizon, :]
        nq = full.shape[-1]
        lv = self.levels if len(self.levels) == nq else list(np.linspace(0.001, 0.999, nq))
        return Pred(point=full[:, :, self._median_index(lv)], quantiles=to_levels(full, lv),
                    meta={"rollout_steps": int(np.ceil(horizon / max(step, 1))),
                          "note": "median-rollback rollout for horizons > one output patch"})

    def trainable(self):
        return None


def _triton_ok() -> tuple[bool, str]:
    """TiRex-2's mLSTM Triton kernels need a recent Triton *and* a recent GPU.

    Measured on this project: Triton 3.1 on an RTX Ada (SM 8.9) dies with
    "PTX compiled with an unsupported toolchain" / missing ninja, so the native
    backend is used there; a Blackwell-class card (SM >= 10.0) with Triton >= 3.4
    runs the kernels as shipped.
    """
    import torch
    try:
        import triton
        tv = tuple(int(x) for x in triton.__version__.split(".")[:2])
    except Exception as e:                                           # noqa: BLE001
        return False, f"triton not importable ({type(e).__name__})"
    if tv < (3, 4):
        return False, f"triton {triton.__version__} < 3.4"
    if not torch.cuda.is_available():
        return False, "no cuda device"
    cap = torch.cuda.get_device_capability()
    if cap < (10, 0):
        return False, f"GPU sm{cap[0]}.{cap[1]} below sm10.0"
    return True, f"triton {triton.__version__}, sm{cap[0]}.{cap[1]}"


def _nvcc() -> str | None:
    """nvcc, even when it is only in a toolkit directory and not on PATH.

    The native mLSTM fallback JIT-compiles through torch's cpp_extension, which looks at
    CUDA_HOME; a CUDA install that is present but not on PATH therefore looked like a
    machine gap. Reporting that as a model failure is wrong in both directions.
    """
    import glob
    import os
    import shutil
    p = shutil.which("nvcc")
    if p:
        return p
    cands = ([os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")]
             if os.environ.get("CUDA_HOME") else [])
    cands += sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            os.environ.setdefault("CUDA_HOME", os.path.dirname(os.path.dirname(c)))
            return c
    return None


def _force_native_mlstm_kernels():
    """Point the vendor's mLSTM backend at the native (non-Triton) kernels.

    Returns an undo callable: the patch is process-wide, so an abandoned fallback would
    otherwise decide how every later task in the same process behaves — which turns one
    failed kernel into a row where every column blames the same wrong thing.
    """
    import tirex2.model.component.mlstm_block as MB
    orig = MB.mLSTMBackendConfig

    def native(*a, **k):
        k.update(chunkwise_kernel="chunkwise--native_autograd",
                 sequence_kernel="native_sequence__native", step_kernel="native",
                 autocast_kernel_dtype="float32")
        return orig(*a, **k)

    MB.mLSTMBackendConfig = native

    def undo():
        MB.mLSTMBackendConfig = orig
        return orig
    return undo


class TiRex2(TiRex):
    key, family = "tirex-2", "tirex2"
    caps = frozenset({"forecast", "quantiles", "covariates"})

    def __init__(self, checkpoint: str = "NX-AI/TiRex-2", kernels: str = "auto", **kw):
        TSFMModel.__init__(self, **kw)
        import os
        if kernels == "auto":
            ok, why = _triton_ok()
            kernels = "triton" if ok else "native"
            self.kernel_reason = why
        if kernels == "native":
            os.environ["USE_TRITON_KERNELS"] = "0"
            if self.device != "cpu" and not _nvcc():
                # the native mLSTM backend JIT-compiles a CUDA extension through
                # flashrnn: without nvcc (a runtime-only CUDA install) neither the
                # Triton kernels nor the native kernels can load. Say so plainly —
                # this is a machine gap, not a model failure.
                from ...errors import UnavailableError
                raise UnavailableError(
                    "tirex-2",
                    f"needs either the Triton kernels ({self.kernel_reason}) or a CUDA "
                    f"toolchain for the native fallback (nvcc not found)",
                    "run it on a Blackwell-class GPU (sm >= 10.0) with CUDA 13 + Triton "
                    ">= 3.4, or install the full CUDA toolkit (nvcc) for the native path")
        import tirex2
        self.checkpoint = checkpoint
        self.kernels = kernels
        if kernels == "native":
            _force_native_mlstm_kernels()
        self.model = tirex2.load_model(ckpt_path=checkpoint, device=self.device)
        if kernels == "auto":
            try:
                self._probe()
            except Exception:                                  # noqa: BLE001
                _force_native_mlstm_kernels()
                self.model = tirex2.load_model(ckpt_path=checkpoint, device=self.device)
                self.kernels = "native"
                self.kernel_reason = "triton path failed its own probe"
        self.max_context = int(getattr(self.model, "max_context_length", 2048))
        self.levels = TiRex._levels(self)

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        """One runtime fallback: the Triton mLSTM scan kernels are the failure mode, and
        whether they compile depends on the GPU arch and the toolchain, not on the model.

        The constructor probe (128 points) can pass while a real batch fails to compile,
        which used to be reported as a TiRex-2 *model* failure. Retrying on the native
        kernels answers the benchmark question instead: does this architecture run here.
        """
        import os
        try:
            return self._predict_impl(contexts, horizon, panels=panels, future=future)
        except Exception as e:                                  # noqa: BLE001
            msg = f"{type(e).__name__} {e}".lower()
            if self.kernels != "triton" or not any(
                    k in msg for k in ("compilationerror", "triton", "ptxas",
                                       "no kernel image", "cuda extension", "ninja")):
                raise
            if not _nvcc():          # native mLSTM JIT-compiles; nothing to fall back to
                raise
            prev = os.environ.get("USE_TRITON_KERNELS")
            os.environ["USE_TRITON_KERNELS"] = "0"
            undo = _force_native_mlstm_kernels()
            import tirex2
            self.model = tirex2.load_model(ckpt_path=self.checkpoint, device=self.device)
            self.kernels = "native"
            self.kernel_reason = (f"triton kernels compiled at import but failed at "
                                  f"runtime ({type(e).__name__}); native kernels tried")
            try:
                return self._predict_impl(contexts, horizon, panels=panels, future=future)
            except Exception as e2:                             # noqa: BLE001
                # leave the process as we found it: an abandoned fallback must not decide
                # how the *next* task in this process behaves
                if prev is None:
                    os.environ.pop("USE_TRITON_KERNELS", None)
                else:
                    os.environ["USE_TRITON_KERNELS"] = prev
                undo()
                try:                    # rebuild the model under the kernels we left
                    self.model = tirex2.load_model(ckpt_path=self.checkpoint,
                                                   device=self.device)
                except Exception:                                  # noqa: BLE001
                    pass
                self.kernels = "triton"
                # the original error is the finding; the retry explains why there is no
                # way around it here. Swapping them would blame the wrong thing.
                raise RuntimeError(
                    f"tirex-2 Triton kernels failed to compile ({type(e).__name__}: "
                    f"{str(e)[:200]}); the native fallback needs a working JIT build and "
                    f"failed too ({type(e2).__name__}: {str(e2)[:200]})") from e

    def _probe(self):
        import torch
        from tirex2.model.types import TimeseriesType
        cs = [np.linspace(0.0, 1.0, 128)]
        batch = [TimeseriesType(target=torch.tensor(c, dtype=torch.float32, device=self.device),
                                past_covariates=None, future_covariates=None) for c in cs]
        self.model.forecast(batch, prediction_length=8, output_type="numpy", batch_size=1)

    def _batch(self, contexts):
        import torch
        from tirex2.model.types import TimeseriesType
        cs = self.trim(contexts)
        return [TimeseriesType(target=torch.tensor(c, dtype=torch.float32, device=self.device),
                               past_covariates=None, future_covariates=None) for c in cs]

    def _predict_impl(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        out = self.model.forecast(self._batch(contexts), prediction_length=int(horizon),
                                  output_type="numpy", batch_size=self.batch)
        q = np.asarray(out, float)
        if q.ndim == 2:
            return Pred(point=q[:, :horizon], meta={"kernels": self.kernels})
        if q.shape[1] != horizon:
            q = np.transpose(q, (0, 2, 1))
        q = q[:, :horizon, :]
        nq = q.shape[-1]
        lv = self.levels if len(self.levels) == nq else list(np.linspace(0.001, 0.999, nq))
        return Pred(point=q[:, :, self._median_index(lv)], quantiles=to_levels(q, lv),
                    meta={"kernels": self.kernels})
