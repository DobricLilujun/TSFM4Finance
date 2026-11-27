"""Rolling-origin evaluation — the part that answers "did fine-tuning help?".

Windows come **only from the evaluation span**, and the metric set is the same for
zero-shot and adapted models, so a delta is a delta.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from ..data.series import SeriesDataset, windows
from ..errors import DataError
from .metrics import classification_report, forecast_report


def _fit_ctx(model, ctx: int, span_len: int, h: int) -> tuple[int, str]:
    """Largest usable context for this span, honouring the model's alignment step."""
    step = int(getattr(model, "ctx_step", 1) or 1)
    if ctx + h <= span_len:
        return int(ctx), ""
    fit = span_len - h
    if step > 1:
        fit = (fit // step) * step
    if fit < max(2 * step, 8):
        raise DataError(f"span holds {span_len} points: too short for horizon {h} "
                        f"with any context a {getattr(model, 'key', '?')} accepts "
                        f"(context step {step})")
    print(f"note: context {ctx} doesn't fit a {span_len}-point span; "
          f"evaluating at context {fit}", file=sys.stderr)
    return int(fit), f"clamped from {ctx} to {fit} (span {span_len}, horizon {h})"


def _finite_or_none(v):
    """Metrics go out as ``None``, never as ``float('nan')``.

    A NaN metric is worse than a blank one: ``json.dumps`` writes a bare ``NaN`` that a
    strict JSON reader rejects, pandas skips it silently while a hand-written ranking
    script (mine, once) treats it as a number and reports a median of NaN, and it hides
    *why* the cell is empty. ``None`` says "no measurement here" and the row's error field
    says why.
    """
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def forecast_eval(model, dataset, *, horizons=(21,), spans=("test",), ctx: int | None = None,
                  stride: int | None = None, max_windows: int | None = 64,
                  batch: int = 16, seed: int = 0, season: int = 1) -> list[dict]:
    ds = dataset if isinstance(dataset, SeriesDataset) else SeriesDataset.load(dataset)
    ctx = int(ctx or min(model.max_context, 256))
    # Honour a checkpoint's declared context. TTM r3 (declared 512) fed 256 padded itself
    # with zeros and returned a forecast at half the level: an honest-looking MASE of 13x
    # naive that was purely our window choice. The protocol says "ctx 256"; for a model
    # that cannot represent 256 the contract wins, and the row records the ctx it was
    # actually scored at in the same column every other row reports.
    mn = int(getattr(model, "min_context", 0) or 0)
    contract_note = ""
    if mn and ctx < mn:
        want = min(mn, int(getattr(model, "max_context", mn) or mn))
        contract_note = f"context raised {ctx}->{want} (checkpoint contract)"
        ctx = want
    rows = []
    for span in spans:
        a, b = ds.splits.get(span, (0, len(ds.values)))
        y_span = ds.values[a:b]
        train = ds.split("train")
        for h in horizons:
            # A context longer than the span is a normal condition in a cross-family
            # sweep (Kairos wants 2048, the built-in fixture has 135 test points), so it
            # degrades instead of raising. Raising here meant the default
            # `tsfm-trl forecast` demo could not run at all on its own fixture.
            eff_ctx, ctx_note = _fit_ctx(model, ctx, len(y_span), h)
            # a covariate panel must come from the SAME offsets as the context. windows()
            # returns the offsets; ds.panel_window() then rebuilds the slice *including*
            # the target column (the panel frame stores open/high/low/volume only, and a
            # raw slice is missing exactly the column panel models ask for).
            want_panel = ds.panel is not None and "panel_in" in set(getattr(model, "caps", ()))
            if "panel_in" in set(getattr(model, "caps", ())) and ds.panel is None:
                # a covariate-fed model cannot be scored on a one-column series; say so
                # as a dataset limitation instead of letting the vendor error stand
                raise DataError(
                    f"{getattr(model, 'key', '?')} consumes a covariate panel, and "
                    f"'{ds.name}' has only the target column: score it on a series with "
                    f"OHLCV covariates (this is a dataset-coverage gap, not a model "
                    f"failure)")
            w = windows(y_span, h, eff_ctx, stride=stride or h, max_windows=max_windows,
                        seed=seed, with_offsets=want_panel)
            if not w:
                raise DataError(f"{ds.name}/{span}: no evaluation windows "
                                f"(need >= {eff_ctx + h}, have {len(y_span)})")
            offs = [x[2] for x in w] if want_panel else None
            cs = [x[0] for x in w]
            ys = np.stack([x[1] for x in w])
            panels = [ds.panel_window(a + o, a + o + eff_ctx) for o in offs] if want_panel else None
            t0 = time.time()
            pred = model.predict(cs, h, panels=panels)
            # A model can emit NaN/Inf for some windows without raising. TTM r3 does this
            # intermittently: the same window was clean in one process and all-NaN in the
            # next. Retry the affected windows singly (a batch can poison a model's own
            # per-instance normalisation), then count what is still broken.
            point = np.asarray(pred.point, dtype=float).reshape(len(cs), -1)
            bad = [i for i in range(len(cs)) if not np.isfinite(point[i]).all()]
            for i in bad:
                try:
                    one = model.predict([cs[i]], h, panels=[panels[i]] if panels else None)
                    cand = np.asarray(one.point, dtype=float).reshape(1, -1)[0]
                    if np.isfinite(cand).all():
                        point[i] = cand
                except Exception:  # noqa: BLE001 - a retry that raises is still a NaN window
                    pass
            if bad:
                pred.point = point
            # A NaN window must not silently shrink the denominator. forecast_report is
            # nan-safe: it would score the surviving 3 of 4 windows while the row still
            # claimed n_windows=4. So the broken windows leave the scoring set, the row
            # reports the denominator that was actually used, and what was dropped is said
            # out loud.
            point = np.asarray(pred.point, dtype=float).reshape(len(cs), -1)
            kept = [i for i in range(len(cs)) if np.isfinite(point[i]).all()]
            dropped = len(cs) - len(kept)
            if dropped:
                ys = np.asarray(ys)[kept]
                pred.point = point[kept]
                for name in ("quantiles", "samples"):
                    arr = getattr(pred, name, None)
                    if arr is not None:
                        setattr(pred, name, np.asarray(arr, dtype=float)[kept])
                pred.meta = {**(pred.meta or {}), "windows_dropped_nonfinite": dropped}
            rep = forecast_report(ys, pred, train, season=season)
            row = {"model": getattr(model, "key", "?"), "adaptation":
                   getattr(model, "adaptation", "zero"), "series": ds.name, "span": span,
                   "horizon": int(h), "ctx": eff_ctx, "n_windows": len(kept),
                   # which bytes and which split the cell was scored on. MASE is divided by
                   # the naive error of the train split, so a CSV (synthetic 70/15/15) and
                   # the arena directory of the same series give different numbers for the
                   # identical forecast; without these two fields that reads as a model or
                   # hardware difference.
                   "data_kind": (ds.meta or {}).get("data_kind"),
                   "data_hash": (ds.meta or {}).get("data_hash"),
                   "seconds": round(time.time() - t0, 2),
                   "no_leakage": True,
                   "ctx_note": "; ".join(x for x in (ctx_note, contract_note) if x),
                   # a NaN in a metric column reads like a number; None plus an error reads
                   # like the missing cell it is
                   **{k: (_finite_or_none(v) if isinstance(v, (int, float)) and
                          not isinstance(v, bool) else v) for k, v in rep.items()}}
            if dropped:
                row["pred_nan_windows"] = dropped
                row["error"] = (f"model emitted non-finite forecasts in {dropped}/{len(cs)} "
                                "windows (after a single-window retry); those windows are "
                                "excluded, so n_windows is the scored count")
            row["meta"] = _scalar_meta(pred.meta)
            rows.append(row)
    return rows


def _scalar_meta(meta) -> dict:
    """Adapter meta that is worth a row: scalars only.

    This is how an adapter gets to state what it measured (MOMENT's `head_from_checkpoint`
    / `head_trained_here`, a clamped context, a caller-constructed interval) instead of
    leaving the table to infer it from the model name. Lists and arrays stay out because a
    markdown cell is not a place for them.
    """
    return {k: v for k, v in (meta or {}).items()
            if isinstance(v, (int, float, str, bool)) and not isinstance(v, type(None))}


def classify_eval(model, cds, *, routes=("probe_stat", "probe_model"), kinds=("logistic",),
                  folds_mode: str = "expanding", seed: int = 0) -> list[dict]:
    """Shallow-classifier classification rows (the probe routes). The end-to-end
    route lives in :class:`tsfm_trl.training.classify.ClassificationTrainer`."""
    from ..probing.features import folds, make_classifier, model_matrix, stat_matrix
    X, y, tr, te = cds.parts()
    rows = []
    for route in routes:
        F = stat_matrix(X) if route == "probe_stat" else model_matrix(model, X)
        for kind in kinds:
            preds = np.full(len(y), -1, dtype=int)
            scores = np.zeros((len(y), max(2, cds.n_classes)))
            for tri, tei in folds(cds.split, mode=folds_mode):
                if len(np.unique(y[tri])) < 2:
                    continue
                clf = make_classifier(kind, seed)
                clf.fit(F[tri], y[tri])
                if hasattr(clf, "predict_proba"):
                    pr = clf.predict_proba(F[tei])
                    scores[tei] = pr if pr.shape[1] == scores.shape[1] else \
                        np.c_[pr, np.zeros((len(tei), scores.shape[1] - pr.shape[1]))]
                preds[tei] = clf.predict(F[tei])
            keep = preds >= 0
            if not keep.any():
                continue
            rep = classification_report(y[keep], preds[keep], n_classes=cds.n_classes)
            pos = None
            if cds.n_classes == 2 and scores.shape[1] == 2:
                pos = scores[keep][:, 1]
                rep.update(classification_report(y[keep], pos, n_classes=2))
            rows.append({"model": getattr(model, "key", "?"), "route": route,
                         "classifier": kind, "dataset": cds.name,
                         "folds": folds_mode, **rep,
                         "n_scored": int(keep.sum())})
    return rows
