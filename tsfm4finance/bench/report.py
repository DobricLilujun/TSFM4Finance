"""Render reports/02_timesfm_benchmark.md from reports/timesfm_benchmark_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tsfm4finance.bench import tasks as T
from tsfm4finance.bench import protocol as P

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "02_timesfm_benchmark.md"
RESULTS = ROOT / "reports" / "timesfm_benchmark_results.json"

DOMAIN_OF = dict(T.REAL_SERIES)
GROUPS = {"index": "Equity indices", "crypto": "Crypto", "fx": "FX",
          "commodity": "Commodities", "bond": "Bonds"}


def _f(x, nd=3):
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:,.1f}"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _agg(res: dict, model: str, h: int, key: str = "mase") -> float:
    vals = []
    for base, _d in T.REAL_SERIES:
        e = res["forecast"].get(f"{base}:h{h}:{model}")
        if e and "aggregate" in e and np.isfinite(e["aggregate"].get(key, np.nan)):
            vals.append(e["aggregate"][key])
    return float(np.mean(vals)) if vals else float("nan")


def build(res: dict | None = None) -> Path:
    res = res or json.loads(RESULTS.read_text())
    meta = res.get("meta", {})
    L: list[str] = []
    A = L.append

    A("# Google TimesFM on real financial series — zero-shot, few-shot, fine-tuned, "
      "classification, anomaly and quant-trading evaluation\n")
    A("Protocol: **spec §4.1** — rolling-origin backtest, non-overlapping windows "
      "(stride = horizon), expanding context capped at 512, MASE scale from the train "
      "split (season = 1), horizons h ∈ {%s} business days.\n"
      % ", ".join(str(h) for h in meta.get("horizons", P.HORIZONS)))
    A("Model: **Google TimesFM 2.5-200M** (`timesfm` 2.0.2 pip, PyTorch checkpoint "
      "`google/timesfm-2.5-200m-pytorch`), full fine-tuning (all 200M parameters). "
      "Reference TSFM baseline: **Chronos-2** (`amazon/chronos-2`); statistical baselines: "
      "naive/persistence, seasonal-naive (week), moving average, ARIMA(1,1,0).\n")
    A("Data: the arena's **12 real yfinance series** (2023-08 → 2026-08 daily closes) plus "
      "the 48 real task datasets built from them (`*_dir5`, `*_volreg`, `*_extreme`, "
      "`*_trade`). Synthetic datasets and the empty `priv_*` stubs are excluded.\n")

    # ---------------- 1. forecasting ----------------#
    A("## 1. Forecasting (zero-shot)\n")
    A("Mean across the 12 real series of the per-series rolling-average metric "
      "(lower MASE / sMAPE / ND is better; ND = MAE relative to seasonal-naive).\n")
    models = ["naive", "seasonal_naive", "moving_average", "arima", "chronos",
              "timesfm@512", "timesfm@256", "timesfm@128"]
    rows = []
    for m in models:
        r = [m]
        for h in (1, 5, 12, 21, 63):
            r.append(_f(_agg(res, m, h, "mase"), 3))
        rows.append(r)
    A(_tbl(["model"] + [f"MASE h={h}" for h in (1, 5, 12, 21, 63)], rows))
    A("")
    A("Dispersion at the primary horizon h=21 (across the 12 series / mean across-window sd):\n")
    drows = []
    for m in models:
        vals = [res["forecast"][f"{b}:h21:{m}"]["aggregate"]["mase"] for b, _d in T.REAL_SERIES
                if f"{b}:h21:{m}" in res["forecast"]]
        wsds = [res["forecast"][f"{b}:h21:{m}"]["aggregate"].get("mase_std", float("nan"))
                for b, _d in T.REAL_SERIES if f"{b}:h21:{m}" in res["forecast"]]
        drows.append([m, _f(np.nanmean(vals), 3), _f(np.nanstd(vals), 3), _f(np.nanmean(wsds), 3)])
    A(_tbl(["model", "mean MASE", "sd across series", "mean across-window sd"], drows))
    A("")
    A("### Per-asset MASE at the primary horizon h=21\n")
    rows = []
    for base, dom in T.REAL_SERIES:
        r = [f"`{base}` ({dom})"]
        for m in models:
            e = res["forecast"].get(f"{base}:h21:{m}")
            if e and "aggregate" in e:
                # mean over rolling windows +- standard deviation over windows
                r.append(f"{_f(e['aggregate']['mase'], 3)} ± {_f(e['aggregate'].get('mase_std'), 3)}")
            else:
                r.append("—")
        rows.append(r)
    A(_tbl(["series"] + models, rows))
    A("\n*Cell = mean ± sd of MASE over the non-overlapping rolling windows "
      "(the JSON also stores sd for every metric as `*_std`).*")
    A("")
    A("### Normalized error vs the persistence forecast on identical windows "
      "(NDp = MAE(model)/MAE(naive); <1 means it beats 'keep the last price')\n")
    rows = []
    for m in models:
        r = [m]
        for h in (1, 5, 12, 21, 63):
            vals = [res["derived"].get(f"{b}:h{h}", {}).get(m, float("nan"))
                    for b, _d in T.REAL_SERIES]
            r.append(_f(np.nanmean(vals), 3))
        rows.append(r)
    A(_tbl(["model"] + [f"NDp h={h}" for h in (1, 5, 12, 21, 63)], rows))
    A("")
    A("### Context ablation (TimesFM zero-shot, mean over 12 series)\n")
    rows = []
    for h in (1, 5, 12, 21, 63):
        rows.append([f"h={h}"] + [_f(_agg(res, f"timesfm@{c}", h, "mase"), 3) for c in (128, 256, 512)]
                    + [_f(_agg(res, f"timesfm@512", h, "nd"), 3),
                       _f(_agg(res, "timesfm@512", h, "crps"), 2),
                       _f(_agg(res, "timesfm@512", h, "violation_rate"), 3)])
    A(_tbl(["horizon", "ctx128", "ctx256", "ctx512", "ND@512", "CRPS@512",
            "10% band violation (target 0.10)"], rows))
    A("")

    # ---------------- 2. fine-tuning ----------------#
    A("## 2. Full fine-tuning (few-shot ladder → full train split)\n")
    A(f"All-parameter fine-tuning of the 200M module, lr={meta.get('ft_lr')}, "
      f"context={meta.get('ft_context')}, target = the model's own 128-step output patch, "
      "loss = pinball over levels 0.1–0.9 + L1 on the median channel. Training windows come "
      "from the **train split only**. Zero-shot reference is the **matched-context** run "
      "(ctx=256).\n")
    rows = []
    for base, dom in T.REAL_SERIES:
        z = res["forecast"].get(f"{base}:h21:timesfm_ft:fs16:{base}")
        r = [f"`{base}`"]
        e = res["forecast"].get(f"{base}:h21:timesfm@256")
        r.append(_f(e["aggregate"]["mase"], 3) if e and "aggregate" in e else "—")
        for tag in ("fs16", "fs64", "full"):
            e = res["forecast"].get(f"{base}:h21:timesfm_ft:{tag}:{base}")
            r.append(_f(e["aggregate"]["mase"], 3) if e and "aggregate" in e else "—")
        ft = res["finetune"].get(f"{base}:{tag}")
        r.append(_f(ft["final_loss"], 3) if isinstance(ft, dict) and "final_loss" in ft else "—")
        rows.append(r)
    A(_tbl(["series", "zero-shot ctx256", "FT K=16", "FT K=64", "FT full", "FT final loss"], rows))
    A("")
    zs = np.mean([_agg(res, "timesfm@256", h, "mase") for h in (5, 21)])
    rows = []
    for tag in ("fs16", "fs64", "full"):
        v = np.nanmean([_agg(res, f"timesfm_ft:{tag}:{b}", h, "mase")
                        for b, _d in T.REAL_SERIES for h in (5, 21)])
        rows.append([f"full fine-tune, K={tag}", _f(v, 3), _f(v - zs, 3)])
    rows.insert(0, ["zero-shot (matched ctx=256)", _f(zs, 3), "—"])
    A("Mean MASE over the 12 series at h ∈ {5, 21}:\n")
    A(_tbl(["setting", "MASE", "Δ vs zero-shot"], rows))
    A("")

    # ---------------- 3. classification ----------------#
    A("## 3. Classification on real labels\n")
    A("`*_dir5`: sign of the next 5-business-day return; `*_volreg`: tercile of the next "
      "21-day realized vol (cut-points fitted on train). Macro-F1 (mean over the 12 series).\n")
    methods = ["majority", "stat_logistic", "stat_gbm", "timesfm_forecastfeat_logistic",
               "timesfm_forecastfeat_gbm", "timesfm_probe_logistic", "timesfm_probe_gbm"]
    for suffix in ("dir5", "volreg"):
        rows = []
        for m in methods:
            vals = [res["classify"][f"{b}_{suffix}"][m]["f1"]
                    for b, _d in T.REAL_SERIES
                    if f"{b}_{suffix}" in res["classify"] and m in res["classify"][f"{b}_{suffix}"]]
            acc = [res["classify"][f"{b}_{suffix}"][m].get("accuracy", float("nan"))
                   for b, _d in T.REAL_SERIES
                   if f"{b}_{suffix}" in res["classify"] and m in res["classify"][f"{b}_{suffix}"]]
            rows.append([m, _f(np.nanmean(vals), 3), _f(np.nanmean(acc), 3), len(vals)])
        A(f"### {suffix}\n")
        A(_tbl(["method", "macro-F1", "accuracy", "n series"], rows))
        A("")

    # ---------------- 4. anomaly ----------------#
    A("## 4. Extreme-move anomaly detection (real labels, expanding MAD, no look-ahead)\n")

    def _nm(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for m in ("oracle_label_rule", "stat_zscore", "timesfm_bandviolation", "timesfm_probe"):
        sel = [res["anomaly"][f"{b}_extreme"].get(m) for b, _d in T.REAL_SERIES
               if f"{b}_extreme" in res["anomaly"]]
        sel = [d for d in sel if isinstance(d, dict) and "f1" in d]
        rows.append([m, _f(_nm([d.get("f1") for d in sel]), 3),
                     _f(_nm([d.get("auc") for d in sel]), 3),
                     _f(_nm([d.get("recall") for d in sel]), 3),
                     len(sel)])
    A(_tbl(["method", "macro-F1", "AUC-ROC", "recall", "n series"], rows))
    A("\n`oracle_label_rule` reproduces the label rule itself and is reported only as a "
      "reference point, not as a competitor; the fair comparisons are `stat_zscore` "
      "(rolling mean/sigma rule, threshold fitted on train) and the TSFM detectors. "
      "`n series` < 12 means some series have no extreme event in the evaluation span.\n")
    A("")

    # ---------------- 5. trading ----------------#
    A("## 5. Quant-trading task (spec §2.3)\n")
    A("Signal = predicted h-day log return from the TSFM point forecast; positions "
      "`sign` (±1) and `voltarget` (clip(s/σ̂,−1,1)); non-overlapping 5-day holding periods on "
      "the test span; costs 0/10/30 bps one-way on |Δposition|.\n")
    rows = []
    for base, dom in T.REAL_SERIES:
        e = res["trading"].get(f"{base}_trade")
        if not e or "model" not in e:
            continue
        sg = e["model"]["sign"]["10bps"]
        vt = e["model"]["voltarget"]["10bps"]
        bh = e["asset_buy_hold"]["10bps"]
        mom = e["benchmarks"]["momentum"]["10bps"]
        rows.append([f"`{base}`", _f(sg["sharpe_net"], 2), _f(vt["sharpe_net"], 2),
                     _f(bh["sharpe_net"], 2), _f(mom["sharpe_net"], 2),
                     _f(sg.get("ic"), 3), _f(sg.get("hit_rate"), 3),
                     _f(sg.get("turnover_total"), 2)])
    A(_tbl(["series", "TSFM sign", "TSFM vol-target", "buy&hold", "momentum",
            "IC", "hit rate", "turnover"], rows))
    A("")
    A("Cost sensitivity (mean net Sharpe over the 12 series):\n")
    rows = []
    for label, path in (("TSFM sign", ("model", "sign")), ("TSFM vol-target", ("model", "voltarget")),
                        ("TSFM FT(K=16) sign", ("model_fs16", "sign")),
                        ("TSFM FT(K=64) sign", ("model_fs64", "sign")),
                        ("TSFM FT(full) sign", ("model_full", "sign")),
                        ("TSFM FT(K=64) vol-target", ("model_fs64", "voltarget")),
                        ("buy & hold", ("asset_buy_hold", None)),
                        ("momentum", ("benchmarks", "momentum"))):
        vals = []
        for cb in (0, 10, 30):
            v = []
            for b, _d in T.REAL_SERIES:
                e = res["trading"].get(f"{b}_trade", {})
                node = e.get(path[0])
                if not node:
                    continue
                node = node.get(path[1]) if path[1] else node
                m = node.get(f"{cb}bps") if isinstance(node, dict) else None
                if m:
                    v.append(m.get("sharpe_net", float("nan")))
            vals.append(_f(np.nanmean(v), 2) if v else "—")
        rows.append([label] + vals)
    A(_tbl(["strategy", "0 bps", "10 bps", "30 bps"], rows))
    A("")

    # ---------------- 6. protocol / sanity ----------------#
    A("## 6. Protocol checks and caveats\n")
    cov = [v["info"]["coverage"] for v in res.get("forecast", {}).values() if "info" in v]
    leak = [v["info"]["no_leakage"] for v in res.get("forecast", {}).values() if "info" in v]
    A(f"- Test-span coverage of the rolling windows: mean **{_f(np.nanmean(cov) if cov else float('nan'), 3)}**, "
      f"min {_f(np.nanmin(cov) if cov else float('nan'), 3)}; no-leakage assertion "
      f"{'held for every window set' if leak and all(leak) else 'FAILED somewhere'}.")
    A("- MASE/ND scale uses the train split only; contexts always end strictly before the "
      "truth block, so no future information enters a forecast.")
    A("- Predictive-band calibration is reported as the 10% quantile violation rate "
      "(target 0.10); a TSFM whose bands are too narrow is misleading for risk use even when "
      "its point forecast is competitive.")
    A("- Trading significance: one asset at one horizon is breadth = 1 (Grinold), and this "
      "grid tests many configurations; per-strategy t-statistics are in the JSON "
      "(`t_stat_net`) and should be read with the multiple-testing caveat "
      "(Harvey 2017 / López de Prado 2018). No borrow cost, slippage, or funding spread is "
      "modelled; |position| ≤ 1, single asset.")
    A("- Fine-tuning regime: three years of a single daily series is a very small, very noisy "
      "corpus for a 200M-parameter model; the K-ladder and LR pilot in §2 quantify what full "
      "fine-tuning costs in out-of-sample accuracy.\n")

    REPORT.write_text("\n".join(L))
    return REPORT


if __name__ == "__main__":
    print(build())
