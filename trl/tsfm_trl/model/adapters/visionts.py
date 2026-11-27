"""VisionTS++ (MIT): MAE-style model that maps time-series patches through a
frozen vision backbone. Weights are raw ``.ckpt`` files (``Lefei/VisionTSpp``) and
the forward lives in the upstream repo, so this adapter is intentionally thin:
it imports the upstream package when present and otherwise reports
``unavailable`` with the exact thing to install — no half-working fake.
"""
from __future__ import annotations

from ...errors import UnavailableError
from ..base import Pred, TSFMModel


class VisionTSpp(TSFMModel):
    key, family = "visiontspp-base", "visionts"
    caps = frozenset({"forecast"})
    loss_kind = None
    max_context = 2048
    max_horizon = 720

    def __init__(self, checkpoint: str = "Lefei/VisionTSpp", variant: str = "base", **kw):
        super().__init__(**kw)
        self.checkpoint, self.variant = checkpoint, variant
        from ...thirdparty import ensure_on_path, hint
        ensure_on_path()
        try:
            from visiontspp.model.visiontspp.visiontspp_model import VisionTSpp_plus  # noqa: F401
        except Exception as e:                                     # noqa: BLE001
            raise UnavailableError(
                self.key,
                "VisionTS++ needs the upstream package (the Hub repo ships .ckpt weights only): "
                f"{e}",
                f"git clone the VisionTS/VisionTS++ upstream, then weights="
                f"/path/to/visiontspp_base.ckpt. {hint('VisionTS')}") from e
        raise UnavailableError(self.key,
                               "upstream VisionTS++ package is importable but this adapter has "
                               "not been wired to its forward yet (tracked as a contribution)")

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        raise UnavailableError(self.key, "adapter not implemented")
