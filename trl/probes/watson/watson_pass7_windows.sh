#!/bin/bash
# watson pass 7: the 66-series corpus at 32 and 64 windows, to test whether the ranking ties
# survive a bigger window sample.
#
# Why: the published ranking sits entirely at --max-windows 8, and docs/08 measured the cost of
# that: at 8 windows, changing only the seed moves naive's own MASE ~55% (median) and the paired
# model/naive ratio ~12%, so a cell within ~12% of parity is neither a win nor a loss. That makes
# the published ties (chronos-bolt-small vs chronos-2, 61.6% of bootstrap resamples;
# chronos-bolt-base vs chronos-t5-small, 53.8%) suspect: they may be an artifact of drawing eight
# windows instead of thirty-two.
#
# This job re-scores the tie cluster plus four baselines at 32 and at 64 windows on the A6000,
# and gpu7 scores the same series in parallel on a GB10 (bridge job gpu7_060_windows.sh). Two
# machines x two window counts means the sensitivity question gets answered on paired data and
# the cross-machine bit-exactness claim is re-checked at a new protocol.
#
# `n_windows` is part of the protocol signature, so these rows never merge with the 8-window
# corpus: the comparison is per window count (probes/window_sensitivity.py).
#
# Submit: sbatch probes/watson/watson_pass7_windows.sh   (from the harness root)
#SBATCH --job-name=trl-pass7-win
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/lujun.li/pass7_watson_win/slurm_%j.out
set -uo pipefail
R=/home/lujun.li/TSFM4Finance-trl/trl
P=/home/lujun.li/envs/tsfm-trl/core/bin/python
DATA=/home/lujun.li/tsfm_data/standard
OUTDIR=/home/lujun.li/pass7_watson_win
MODELS=naive,seasonal_naive,arima,theta,chronos-bolt-small,chronos-bolt-base,chronos-2,chronos-t5-small
# the same 66 arena series gpu7 scored in pass 4/5, so a cell here has a twin on the GB10
SERIES="std_equity_1d_10107 std_equity_1d_11308 std_equity_1d_12060 std_equity_1d_12490 std_equity_1d_13901 std_equity_1d_13928 std_equity_1d_14277 std_equity_1d_14541 std_equity_1d_17005 std_equity_1d_17830 std_equity_1d_18542 std_equity_1d_19561 std_equity_1d_21936 std_equity_1d_22111 std_equity_1d_22752 std_equity_1d_26403 std_equity_1d_38703 std_equity_1d_43449 std_equity_1d_48725 std_equity_1d_50876 std_equity_1d_59176 std_equity_1d_59328 std_equity_1d_65875 std_equity_1d_66093 std_equity_1d_66181 std_equity_1d_70519 std_equity_1d_77178 std_equity_1d_83443 std_equity_1d_86356 std_equity_1d_86868 std_equity_1d_90319 std_equity_1d_91233 std_equity_1d_92611 std_fi_1d_de_10y std_fi_1d_de_2y std_fi_1d_de_5y std_fi_1d_jp_10y std_fi_1d_jp_2y std_fi_1d_jp_5y std_fi_1d_uk_20y std_fi_1d_uk_2y std_fi_1d_uk_7y std_fi_1d_us_10y std_fi_1d_us_20y std_fi_1d_us_30y std_fi_1d_us_3y std_fi_1d_us_7y std_fx_1d_audusd std_fx_1d_euraud std_fx_1d_eurgbp std_fx_1d_eurjpy std_fx_1d_gbpjpy std_fx_1d_gbpusd std_fx_1d_usdchf std_fx_1d_usdcnh std_fx_1d_usdinr std_fx_1d_usdjpy std_fx_1d_usdpln std_fx_1d_usdtry std_fx_1d_usdzar std_fx_1h_audusd std_fx_1h_euraud std_fx_1h_eurchf std_fx_1h_gbpusd std_fx_1h_nzdusd std_fx_1h_usdjpy"
mkdir -p "$OUTDIR"
cd "$R" || exit 7
timeout 200 git -C "$R" pull -q --ff-only origin trl-integration 2>/dev/null \
  || echo "WARN: repo not advanced (offline or diverged); rows carry the sha printed below"
# a fresh clone has no venvs; without --envs every model returns an env-skip error record
# and the lane file has the same line count as a scored one
[ -x "$P" ] || { echo "FATAL: no core env at $P (run the watson env setup first)"; exit 8; }
"$P" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 9)" || {
  echo "FATAL: core env has no CUDA on this allocation"; exit 9; }
ARGS=()
for s in $SERIES; do
  [ -d "$DATA/$s" ] || { echo "FATAL: no arena dir for $s"; exit 6; }
  ARGS+=(--series "$DATA/$s")
done
echo "-- series: $(echo "$SERIES" | wc -w) arena dirs | models: 8 | horizons 21,63 | windows: 32 then 64"
rc_all=0
for MW in 32 64; do
  OUT=$OUTDIR/lane_win$MW.jsonl
  echo "== windows=$MW -> $OUT"
  "$P" -u scripts/sweep_evaluate.py --envs /home/lujun.li/envs/tsfm-trl ${ARGS[@]} \
    --models "$MODELS" --horizons 21,63 --max-windows "$MW" --seed 0 \
    --device cuda --fresh --out "$OUT"
  rc=$?
  [ $rc -ne 0 ] && rc_all=$rc
  "$P" - "$OUT" <<'SUMMARY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
meta = next((r["meta"] for r in recs if "meta" in r), {})
ok = [r for r in recs if "meta" not in r and not r.get("error")]
cells = sum(len(r.get("rows", [])) for r in ok)
errs = [r["model"] for r in recs if "meta" not in r and r.get("error")]
nw = sorted({row.get("n_windows") for r in ok for row in r.get("rows", [])})
print(f"records={len(recs)} scored={len(ok)} cells={cells} errors={len(errs)} {errs}")
print(f"machine={meta.get('machine')} sha={meta.get('code_sha')} device={meta.get('device')} "
      f"kinds={meta.get('data_kinds')} lane max_windows={meta.get('max_windows')} row n_windows={nw}")
SUMMARY
done
exit $rc_all
