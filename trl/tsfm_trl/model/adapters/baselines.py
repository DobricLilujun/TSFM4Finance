"""Statistical baselines. Not foundation models, but they are what every TSFM row
in a table must be compared against, and they keep the library usable (and testable)
with no torch and no weights installed."""
from __future__ import annotations

import numpy as np

from ...errors import UnavailableError
from ..base import Pred, TSFMModel, quantiles_from_samples


class _Naive(TSFMModel):
    caps = frozenset({"forecast"})
    max_context = 100_000
    max_horizon = 100_000

    def _point(self, cs, h):
        raise NotImplementedError

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        cs = [np.asarray(c, float) for c in contexts]
        return Pred(point=np.stack([self._point(c, horizon) for c in cs])[:, :horizon])

    def trainable(self):
        return None


class Naive(_Naive):
    """Persistence: every future step = last observation."""
    key, family = "naive", "baseline"

    def _point(self, c, h):
        v = c[-1] if len(c) else 0.0
        return np.full(h, v)


class SeasonalNaive(_Naive):
    key, family = "seasonal_naive", "baseline"

    def __init__(self, season: int = 5, **kw):
        super().__init__(**kw)
        self.season = int(season)

    def _point(self, c, h):
        s = self.season
        if len(c) < s:
            return np.full(h, c[-1] if len(c) else 0.0)
        tail = c[-s:]
        return np.tile(tail, int(np.ceil(h / s)))[:h]


class MovingAverage(_Naive):
    key, family = "moving_average", "baseline"

    def __init__(self, window: int = 20, **kw):
        super().__init__(**kw)
        self.window = int(window)

    def _point(self, c, h):
        w = min(self.window, len(c)) or 1
        return np.full(h, float(np.mean(c[-w:])))


class Theta(TSFMModel):
    key, family = "theta", "baseline"
    max_context = 100_000

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        """Theta (Assa-Makridakis 2008). ``statsmodels`` never shipped ``ThetaModel``,
        so the method is implemented here (drift line + short-term component, with θ
        chosen by in-sample MAE on a small grid); if a future statsmodels release
        provides it, that path is preferred."""
        pts = []
        for c in contexts:
            c = np.asarray(c, float)
            if len(c) < 24 or not np.all(np.isfinite(c)):
                pts.append(np.full(horizon, float(c[-1]) if len(c) else 0.0))
                continue
            pts.append(self._theta(c, int(horizon)))
        return Pred(point=np.stack(pts)[:, :horizon],
                    meta={"method": "theta", "reference": "Assa-Makridakis (2008)"})

    @staticmethod
    def _theta(y: np.ndarray, h: int) -> np.ndarray:
        """Theta, the way the M-competitions used it: a linear **drift line** plus a
        simple-exponential-smoothing forecast of the short-term component
        ``se = (2θ−1)·y + (1−2θ)·drift``. θ and the smoothing constant are chosen by
        one-step in-sample MAE — a baseline gets tuned the same way a model would be,
        otherwise "the TSFM beats classical" is not a fair sentence."""
        from scipy.stats import linregress
        n = len(y)
        t = np.arange(n, dtype=float)
        r = linregress(t, y)
        grid = np.arange(n + h, dtype=float)
        drift = r.intercept + r.slope * grid

        def ses(x: np.ndarray, alpha: float):
            """SES level of the short-term component, scored on the **combined**
            one-step forecast (drift + level). Scoring the component alone would let
            θ=0.5 win by construction: there the component is identically zero, so its
            own error is zero while the drift-only forecast is usually terrible."""
            lvl = x[0]
            errs = []
            for i in range(1, n):
                errs.append(abs(y[i] - (drift[i] + lvl)))
                lvl = lvl + alpha * (x[i] - lvl)
            return lvl, float(np.mean(errs))

        best = None
        for theta in (0.5, 0.75, 1.0):
            se = (2 * theta - 1) * y + (1 - 2 * theta) * drift[:n]
            for alpha in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
                lvl, mae = ses(se, alpha)
                if best is None or mae < best[0]:
                    best = (mae, theta, alpha, lvl)
        _, theta, alpha, lvl = best
        return drift[n:] + lvl


class Arima(TSFMModel):
    """ARIMA(p,d,q) refit per origin. Quantiles come from the fitted residual
    sd (a Gaussian interval — stated in ``Pred.meta``, not hidden)."""

    key, family = "arima", "baseline"
    caps = frozenset({"forecast", "quantiles"})
    max_context = 5000

    def __init__(self, order=(1, 1, 0), **kw):
        super().__init__(**kw)
        self.order = tuple(order)

    def predict(self, contexts, horizon, *, panels=None, future=None) -> Pred:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as e:                                   # pragma: no cover
            raise UnavailableError("arima", f"statsmodels missing: {e}", "[baselines]") from e
        pts, paths = [], []
        rng = np.random.RandomState(self.seed)
        for c in contexts:
            c = np.asarray(c, float)
            c = c[np.isfinite(c)]
            if len(c) < 40:
                base = np.full(horizon, float(c[-1]) if len(c) else 0.0)
                pts.append(base)
                paths.append(np.tile(base, (32, 1)))
                continue
            try:
                res = ARIMA(c, order=self.order).fit()
                fc = np.asarray(res.forecast(horizon), float)[:horizon]
                sd = float(np.std(res.resid)) if np.isfinite(res.resid).all() else 0.0
            except Exception:                                      # noqa: BLE001
                fc = np.full(horizon, float(c[-1]))
                sd = float(np.std(np.diff(c))) if len(c) > 1 else 0.0
            pts.append(fc)
            paths.append(fc[None, :] + rng.randn(32, horizon) * sd)
        paths = np.stack(paths)[:, :32, :][:, :, :horizon]
        return Pred(point=np.stack(pts)[:, :horizon],
                    quantiles=quantiles_from_samples(paths),
                    meta={"order": list(self.order), "interval": "gaussian from residual sd"})

    def trainable(self):
        return None
