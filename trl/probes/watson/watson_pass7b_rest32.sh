#!/bin/bash
# watson pass 7b: put the rest of the fleet on the 32-window protocol.
#
# Why: pass 7 re-scored 8 models on these same 66 arena series at 32 and 64 windows and found the
# beat-naive share is a *threshold on a noisy number*: at 8 windows every model's share is inflated
# (chronos-bolt-small 42.4% -> 15.2% at 32), one pairwise ordering flips (chronos-bolt-small >
# chronos-2, 53.8% -> 40.2%), and 33% of cells change winner. The published table is only internally
# consistent if the whole fleet shares one window budget, so the other 11 model keys move to 32 here.
#
# Same 66 series gpu7 scored (not watson's extra 35): a cell with no twin on another machine cannot
# be reproducibility-checked, and this pass exists to make the table comparable.
#
# Submit: sbatch probes/watson/watson_pass7b_rest32.sh   (from the harness root)
#SBATCH --job-name=trl-pass7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/lujun.li/pass7b_watson_32/slurm_%j.out
set -uo pipefail
R=/home/lujun.li/TSFM4Finance-trl/trl
P=/home/lujun.li/envs/tsfm-trl/core/bin/python
DATA=/home/lujun.li/tsfm_data/standard
OUTDIR=/home/lujun.li/pass7b_watson_32
MODELS=ttm-r2,ttm-r3,timesfm-2.5-200m,timesfm-3.0,flowstate-r1,patchtst-fm-r2,moment-1-small,moment-1-large,timemoe-50m,chronos-t5-base,moving_average
SERIES="std_equity_1d_10107 std_equity_1d_11308 std_equity_1d_12060 std_equity_1d_12490 std_equity_1d_13901 std_equity_1d_13928 std_equity_1d_14277 std_equity_1d_14541 std_equity_1d_17005 std_equity_1d_17830 std_equity_1d_18542 std_equity_1d_19561 std_equity_1d_21936 std_equity_1d_22111 std_equity_1d_22752 std_equity_1d_26403 std_equity_1d_38703 std_equity_1d_43449 std_equity_1d_48725 std_equity_1d_50876 std_equity_1d_59176 std_equity_1d_59328 std_equity_1d_65875 std_equity_1d_66093 std_equity_1d_66181 std_equity_1d_70519 std_equity_1d_77178 std_equity_1d_83443 std_equity_1d_86356 std_equity_1d_86868 std_equity_1d_90319 std_equity_1d_91233 std_equity_1d_92611 std_fi_1d_de_10y std_fi_1d_de_2y std_fi_1d_de_5y std_fi_1d_jp_10y std_fi_1d_jp_2y std_fi_1d_jp_5y std_fi_1d_uk_20y std_fi_1d_uk_2y std_fi_1d_uk_7y std_fi_1d_us_10y std_fi_1d_us_20y std_fi_1d_us_30y std_fi_1d_us_3y std_fi_1d_us_7y std_fx_1d_audusd std_fx_1d_euraud std_fx_1d_eurgbp std_fx_1d_eurjpy std_fx_1d_gbpjpy std_fx_1d_gbpusd std_fx_1d_usdchf std_fx_1d_usdcnh std_fx_1d_usdinr std_fx_1d_usdjpy std_fx_1d_usdpln std_fx_1d_usdtry std_fx_1d_usdzar std_fx_1h_audusd std_fx_1h_euraud std_fx_1h_eurchf std_fx_1h_gbpusd std_fx_1h_nzdusd std_fx_1h_usdjpy"
mkdir -p "$OUTDIR"
cd "$R" || exit 7
timeout 200 git -C "$R" pull -q --ff-only origin trl-integration 2>/dev/null \
  || echo "WARN: repo not advanced (offline or diverged); rows carry the sha printed below"
[ -x "$P" ] || { echo "FATAL: no core env at $P (run the watson env setup first)"; exit 8; }
"$P" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 9)" || {
  echo "FATAL: core env has no CUDA on this allocation"; exit 9; }
ARGS=()
for s in $SERIES; do
  [ -d "$DATA/$s" ] || { echo "FATAL: no arena dir for $s"; exit 6; }
  ARGS+=(--series "$DATA/$s")
done
echo "-- series: $(echo "$SERIES" | wc -w) arena dirs | models: 11 | horizons 21,63 | windows 32"
echo "-- sha=$(git -C "$R" rev-parse --short HEAD) torch=$("$P" -c 'import torch;print(torch.__version__)')"
OUT=$OUTDIR/lane32.jsonl
# moment/timemoe live in their own envs; --envs is what makes those keys scoreable at all
"$P" -u scripts/sweep_evaluate.py --envs /home/lujun.li/envs/tsfm-trl "${ARGS[@]}" \
  --models "$MODELS" --horizons 21,63 --max-windows 32 --seed 0 \
  --device cuda --fresh --out "$OUT"
rc=$?
"$P" - "$OUT" <<'SUMMARY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
meta = next((r["meta"] for r in recs if "meta" in r), {})
ok = [r for r in recs if "meta" not in r and r.get("rows")]
bad = [(r.get("model"), str(r.get("error"))[:120]) for r in recs if "meta" not in r and not r.get("rows")]
cells = sum(len(r["rows"]) for r in ok)
print(f"models ok={len(ok)} failed={len(bad)} cells={cells}")
print(f"meta: machine={meta.get('machine')} sha={meta.get('code_sha')} device={meta.get('device')} "
      f"max_windows={meta.get('max_windows')} data_kinds={meta.get('data_kinds')}")
for m, e in bad:
    print(f"  FAILED {m}: {e}")
SUMMARY
exit $rc
