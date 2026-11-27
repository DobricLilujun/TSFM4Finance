#!/bin/bash
# watson pass-6: MOMENT and TimeMoE over the 66-series half, on the arena parquet path.
#
# Why: those two families were scored once, on trux, in pass 3. Neither watson (pass-4/pass-5,
# core env only) nor gpu7 (no moment/timemoe env) has a row for them, so their numbers sit in the
# merged table with no second measurement anywhere - the exact situation pass 5 was built to remove.
# MOMENT additionally carries the untrained-head flag (the public checkpoint ships only the
# reconstruction head; the forecasting head is randomly initialised), so a second machine is the
# cheap way to show that the flag is a property of the checkpoint and not of one run.
#
# Submit: sbatch probes/watson/watson_pass6_moment.sh   (from the harness root)
#SBATCH --job-name=trl-pass6-mt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/lujun.li/pass6_watson_mt/slurm_%j.out
set -uo pipefail
R=/home/lujun.li/TSFM4Finance-trl/trl
P=/home/lujun.li/envs/tsfm-trl/core/bin/python
DATA=/home/lujun.li/tsfm_data/standard
OUT=/home/lujun.li/pass6_watson_mt/lane.jsonl
MODELS=moment-1-small,moment-1-large,timemoe-50m
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
echo "-- series: $(echo "$SERIES" | wc -w) arena dirs | models: $(echo "$MODELS" | tr "," "\n" | wc -l) | horizons 21,63"
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
