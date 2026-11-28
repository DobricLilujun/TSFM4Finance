"""Statistical baselines.  Non-negotiable in finance: Rahimikia et al. show
zero-shot TSFMs losing to classical rules, so a table without them is unreadable."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Engine, Pred


class NaiveEngine(Engine):
    key, family = "naive", "baseline"
    caps = {"forecast"}
    max_context = 512

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        last = np.array([float(np.asarray(c, float)[-1]) for c in contexts])[:, None]
        return Pred(point=np.repeat(last, horizon, axis=1))


class SeasonalNaiveEngine(Engine):
    key, family = "seasonal_naive", "baseline"
    caps = {"forecast"}
    max_context = 512

    def __init__(self, season: int = 5, **kw):
        super().__init__(**kw)
        self.season = int(season)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        pts = []
        for c in contexts:
            c = np.asarray(c, float)
            s = self.season
            if len(c) < s:
                pts.append([float(c[-1])] * horizon)
            else:
                pts.append([float(c[-s:][i % s]) for i in range(horizon)])
        return Pred(point=np.array(pts))


class MovingAverageEngine(Engine):
    key, family = "moving_average", "baseline"
    caps = {"forecast"}
    max_context = 512

    def __init__(self, window: int = 20, **kw):
        super().__init__(**kw)
        self.window = int(window)

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        pts = [float(np.mean(np.asarray(c, float)[-self.window:])) for c in contexts]
        return Pred(point=np.tile(np.array(pts)[:, None], (1, horizon)))


class _StatsmodelEngine(Engine):
    """Shared driver for refit-at-origin statistical fits (ARIMA / ETS / Theta)."""
    family = "baseline"
    caps = {"forecast"}
    max_context = 512

    def __init__(self, **kw):
        super().__init__(**kw)
        self.order = kw.get("order", (1, 1, 0))

    def _fit_predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def predict(self, contexts, horizon, *, panels=None) -> Pred:
        pts = []
        for c in contexts:
            y = np.asarray(c, float)[-self.max_context:]
            try:
                f = self._fit_predict(y, horizon)
            except Exception:                                     # noqa: BLE001
                f = np.repeat(float(y[-1]), horizon)               # fall back to naive
            pts.append(np.asarray(f, float)[:horizon])
        return Pred(point=np.array(pts))


class ArimaEngine(_StatsmodelEngine):
    """Fixed-order ARIMA(p,d,q) refit at every origin.

    The order stays fixed (default (1,1,0)) rather than auto-selected: auto_arima
    on financial prices collapses to (0,1,0) = persistence, which would quietly
    duplicate the naive row.  ``arima_auto`` is the selection-based variant.
    """
    key = "arima"

    def __init__(self, order=(1, 1, 0), **kw):
        super().__init__(**kw)
        self.order = tuple(order)

    def _fit_predict(self, y, horizon):
        import pmdarima as pm
        base = float(y[-1])
        z = y - base
        m = pm.ARIMA(order=self.order, seasonal=False, suppress_warnings=True,
                     method="lbfgs", maxiter=300)
        m.fit(z)
        return np.asarray(m.predict(n_periods=horizon), dtype=float) + base


class ArimaAutoEngine(ArimaEngine):
    key = "arima_auto"

    def _fit_predict(self, y, horizon):
        import pmdarima as pm
        m = pm.auto_arima(y, start_p=0, d=1, start_q=0, max_p=2, max_q=2, m=1,
                          seasonal=False, stepwise=True, suppress_warnings=True,
                          error_action="ignore", max_order=4, trace=False)
        return m.predict(n_periods=horizon)


class EtsEngine(_StatsmodelEngine):
    key = "ets"

    def _fit_predict(self, y, horizon):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(y, trend="add", seasonal=None,
                                 initialization_method="estimated").fit()
        return m.forecast(horizon)


class ThetaEngine(_StatsmodelEngine):
    key = "theta"

    def _fit_predict(self, y, horizon):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        import pandas as pd
        m = ThetaModel(pd.Series(y), period=5).fit()
        return np.asarray(m.forecast(horizon).to_numpy(), dtype=float)
