#!/bin/bash
# watson pass-5: score the 66 series gpu7 (DGX Spark) covered, on the arena parquet path.
#
# Why this job exists: the pass-3/pass-4 fleet deliberately partitioned series so machines
# would not duplicate work - watson took 35, trux/gpu7 took 66, and the two sets do not
# intersect. The consequence is that the cross-machine stratum in a merged 101-series report
# is only 216 cells: there is nothing to compare, not because the numbers disagree but
# because the same cell was never measured twice. This job re-scores gpu7's half on an
# A6000 with a different torch build, which turns the leaderboard's headline claim
# (8 of 9 families reproduce bit-for-bit) from a 12-series stratum into a 66-series one.
#
# Submit: sbatch probes/watson/watson_pass5_66.sh   (from the harness root)
#SBATCH --job-name=trl-pass5-66
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/lujun.li/pass5_watson_66/slurm_%j.out
set -uo pipefail
R=/home/lujun.li/TSFM4Finance-trl/trl
P=/home/lujun.li/envs/tsfm-trl/core/bin/python
DATA=/home/lujun.li/tsfm_data/standard
OUT=/home/lujun.li/pass5_watson_66/lane.jsonl
MODELS=arima,moving_average,naive,seasonal_naive,theta,chronos-2,chronos-bolt-small,chronos-bolt-base,chronos-t5-small,chronos-t5-base,flowstate-r1,timesfm-2.5-200m,timesfm-3.0,ttm-r2,ttm-r3,patchtst-fm-r2
SERIES="std_equity_1d_10107 std_equity_1d_11308 std_equity_1d_12060 std_equity_1d_12490 std_equity_1d_13901 std_equity_1d_13928 std_equity_1d_14277 std_equity_1d_14541 std_equity_1d_17005 std_equity_1d_17830 std_equity_1d_18542 std_equity_1d_19561 std_equity_1d_21936 std_equity_1d_22111 std_equity_1d_22752 std_equity_1d_26403 std_equity_1d_38703 std_equity_1d_43449 std_equity_1d_48725 std_equity_1d_50876 std_equity_1d_59176 std_equity_1d_59328 std_equity_1d_65875 std_equity_1d_66093 std_equity_1d_66181 std_equity_1d_70519 std_equity_1d_77178 std_equity_1d_83443 std_equity_1d_86356 std_equity_1d_86868 std_equity_1d_90319 std_equity_1d_91233 std_equity_1d_92611 std_fi_1d_de_10y std_fi_1d_de_2y std_fi_1d_de_5y std_fi_1d_jp_10y std_fi_1d_jp_2y std_fi_1d_jp_5y std_fi_1d_uk_20y std_fi_1d_uk_2y std_fi_1d_uk_7y std_fi_1d_us_10y std_fi_1d_us_20y std_fi_1d_us_30y std_fi_1d_us_3y std_fi_1d_us_7y std_fx_1d_audusd std_fx_1d_euraud std_fx_1d_eurgbp std_fx_1d_eurjpy std_fx_1d_gbpjpy std_fx_1d_gbpusd std_fx_1d_usdchf std_fx_1d_usdcnh std_fx_1d_usdinr std_fx_1d_usdjpy std_fx_1d_usdpln std_fx_1d_usdtry std_fx_1d_usdzar std_fx_1h_audusd std_fx_1h_euraud std_fx_1h_eurchf std_fx_1h_gbpusd std_fx_1h_nzdusd std_fx_1h_usdjpy"
mkdir -p "$(dirname "$OUT")"
cd "$R" || exit 7
# a fresh clone has no venvs; without --envs every model returns an env-skip error record
# and the lane file has the same line count as a scored one
[ -x "$P" ] || { echo "FATAL: no core env at $P (run probes/watson env setup first)"; exit 8; }
"$P" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 9)" || {
  echo "FATAL: core env has no CUDA on this allocation"; exit 9; }
ARGS=()
for s in $SERIES; do
  [ -d "$DATA/$s" ] || { echo "FATAL: no arena dir for $s"; exit 6; }
  ARGS+=(--series "$DATA/$s")
done
echo "-- series: $(echo "$SERIES" | wc -w) arena dirs | models: 16 | horizons 21,63"
"$P" -u scripts/sweep_evaluate.py --envs /home/lujun.li/envs/tsfm-trl ${ARGS[@]} \
  --models "$MODELS" --horizons 21,63 --max-windows 8 --seed 0 \
  --device cuda --fresh --out "$OUT"
rc=$?
"$P" - "$OUT" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
meta = next((r["meta"] for r in recs if "meta" in r), {})
ok = [r for r in recs if "meta" not in r and not r.get("error")]
cells = sum(len(r.get("rows", [])) for r in ok)
errs = [r["model"] for r in recs if "meta" not in r and r.get("error")]
print(f"records={len(recs)} scored={len(ok)} cells={cells} errors={len(errs)}")
print(f"machine={meta.get('machine')} sha={meta.get('code_sha')} device={meta.get('device')} kinds={meta.get('data_kinds')}")
PYEOF
exit $rc
