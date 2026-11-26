#!/usr/bin/env bash
# Run this ON THE SPARK (gpu7). It is the only side that can dial trux, so the worker
# pulls its command from trux and pushes results back. Nothing listens on the Spark.
#
#   BRIDGE_SSH='ssh -i ~/.ssh/id_ed25519_gm_transfer' BRIDGE_USER=yewei \
#   BRIDGE_HOST=trux-tsubame.uni.lux BRIDGE_DIR=/raid/yewei/spark_bridge \
#   nohup ./spark_worker.sh > ~/spark_worker.log 2>&1 &
#
# One job at a time, sequence-numbered, with a heartbeat that carries this machine's
# environment — cross-machine rows are only meaningful with that context attached.
set -u
BRIDGE_SSH=${BRIDGE_SSH:-"ssh -i $HOME/.ssh/id_ed25519_gm_transfer -o StrictHostKeyChecking=accept-new"}
BRIDGE_USER=${BRIDGE_USER:-yewei}
BRIDGE_HOST=${BRIDGE_HOST:-trux-tsubame.uni.lux}
BRIDGE_DIR=${BRIDGE_DIR:-/raid/yewei/spark_bridge}
POLL=${POLL:-60}
TRANSPORT=${TRANSPORT:-remote}
LOCAL=${LOCAL:-$HOME/spark_bridge}
# TRANSPORT=local exists for testing the control plane on one machine; the real worker
# uses ssh/rsync and is the only side that can dial trux. Both the transport AND the
# path prefix have to switch — leaving `user@host:` on a local rsync silently does
# nothing, which is how a worker looks alive while never transferring a byte.
RS() { if [ "${TRANSPORT:-remote}" = local ]; then rsync -a "$@"; else rsync -e "$BRIDGE_SSH" -a "$@"; fi; }
RDEST() { if [ "${TRANSPORT:-remote}" = local ]; then printf '%s' "$BRIDGE_DIR/$1";
          else printf '%s' "$BRIDGE_USER@$BRIDGE_HOST:$BRIDGE_DIR/$1"; fi; }
mkdir -p "$LOCAL/inbox" "$LOCAL/outbox/results"
# Exactly one poller per bridge directory. Two workers both see the same inbox, and the
"have I already run this job?" check is a file that appears only after a run starts, so
overlapping polls can run one job twice on one GPU. Restarting a worker without killing
the old one is the normal way this happens.
LOCK="$LOCAL/worker.pid"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  old=$(cat "$LOCK"); [ "$old" != "$$" ] && {
    echo "another worker (pid $old) is already polling $LOCAL; kill it first" >&2; exit 1; }
fi
echo $$ > "$LOCK"
py_candidate() {
  # The worker's shell rarely has the venv active: PY is set inside job scripts, not in
  # the poller's environment, and a standalone clone keeps its envs under ./envs. Try the
  # interpreters that plausibly hold torch, in order, and use the first that imports.
  for c in "${PY:-}" "$HOME/envs/tsfm-trl/core/bin/python" "$HOME/projects/TSFM4Finance/.venv/bin/python" \
           "${REPO:-.}/envs/core/bin/python" python3; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && "$c" -c 'import torch' >/dev/null 2>&1 \
      && { echo "$c"; return; }
  done
}
torch_free() {
  local c; c=$(py_candidate)
  [ -z "$c" ] && return
  "$c" -c 'import torch;print(",".join(str(torch.cuda.mem_get_info(i)[0]>>20) for i in range(torch.cuda.device_count())))' 2>/dev/null | tr '\n' ','
}
beat() {                                   # machine facts = provenance, not decoration
  { echo "host=$(hostname)"; echo "arch=$(uname -m)"; echo "when=$(date -u +%FT%TZ)"
    # REPO differs per machine (standalone clone vs vendored trl/ inside TSFM4Finance);
    # an unresolved path used to leave code_sha blank, which is worse than absent because
    # it looks like a clean tree at an unknown commit.
    r="${REPO:-}"
    [ -d "$r/.git" ] || r="$(for c in "$HOME/TSFM4Finance-trl" "$HOME/tsfm-trl" "$HOME/projects/TSFM4Finance-trl"; do
        [ -d "$c/.git" ] && { echo "$c"; break; }; done)"
    [ -n "$r" ] && [ -d "$r/.git" ] &&
      echo "code_sha=$(git -C "$r" rev-parse --short HEAD) branch=$(git -C "$r" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$r" status --porcelain | wc -l)"
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',')
    case "${free//[[:space:]]/}" in
      *[N/]*|"")
        # GB10/DGX Spark reports [N/A] for memory under unified memory; torch still sees
        # the device, so schedule from torch rather than treating the card as absent.
        free=$(torch_free)
        echo "gpu_free=${free} gpu_free_source=torch"
        ;;
      *) echo "gpu_free=${free} gpu_free_source=nvidia-smi" ;;
    esac
    TPY=$(py_candidate)
    echo "torch_py=${TPY:-none}"
    [ -n "$TPY" ] && "$TPY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())' 2>/dev/null
  } > "$LOCAL/outbox/heartbeat" 2>/dev/null
  RS "$LOCAL/outbox/heartbeat" "$(RDEST outbox)/" 2>/dev/null
}
echo "worker up: $BRIDGE_USER@$BRIDGE_HOST:$BRIDGE_DIR poll=${POLL}s"
while :; do
  RS "$(RDEST inbox)/" "$LOCAL/inbox/" 2>/dev/null
  # learn which jobs already have a log: trux keeps the job file in inbox until it
  # reaps it, and rsync will hand it back to us on every poll. Without this check the
  # same job runs forever.
  RS "$(RDEST outbox)/" "$LOCAL/outbox/" 2>/dev/null
  job=""
  for cand in $(ls -1 "$LOCAL/inbox"/*.sh 2>/dev/null | grep -v '\.done\.sh$' | sort); do
    if [ ! -f "$LOCAL/outbox/$(basename "$cand").log" ]; then job=$cand; break; fi
  done
  if [ -n "$job" ]; then
    name=$(basename "$job"); stamp=$(date -u +%FT%TZ)
    echo "running $name at $stamp"
    # the log goes next to the job name so trux can tell a failure from an idle worker
    bash "$job" > "$LOCAL/outbox/$name.log" 2>&1; rc=$?
    mv "$job" "${job%.sh}.done.sh"
    RS "$LOCAL/outbox/" "$(RDEST outbox)/" 2>/dev/null
    RS "$LOCAL/inbox/" "$(RDEST inbox)/" 2>/dev/null
    echo "$name rc=$rc at $stamp"
  fi
  beat
  sleep "$POLL"
done
