#!/usr/bin/env bash
# trux side: enqueue a job for the Spark worker, optionally wait for its log.
#   trux_dispatch.sh job.sh [--wait] [--timeout 5400]
# endpoints (bridge dir, hosts, keys) live in an untracked sibling file; see
# bridge.env.example. Public tree carries no lab hostnames or scratch paths.
_BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$_BRIDGE_DIR/bridge.env" ] && . "$_BRIDGE_DIR/bridge.env"
set -u
BRIDGE=${BRIDGE:?BRIDGE unset - copy bridge.env.example to bridge.env}
job=${1:?job script required}; shift || true
wait=0; timeout_s=5400
while [ $# -gt 0 ]; do case $1 in --wait) wait=1;; --timeout) timeout_s=$2; shift;; esac; shift; done
[ -f "$BRIDGE/outbox/heartbeat" ] || { echo "no heartbeat yet: the worker has not dialed in (or is dead)" >&2; }
if [ -f "$BRIDGE/outbox/heartbeat" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$BRIDGE/outbox/heartbeat") ))
  echo "heartbeat ${age}s old:"; sed 's/^/  /' "$BRIDGE/outbox/heartbeat"
  [ "$age" -gt 900 ] && echo "WARNING: worker silent for ${age}s — results may be stale" >&2
fi
n=$(printf '%03d' $(( $(ls -1 "$BRIDGE/inbox"/*.done.sh 2>/dev/null | wc -l) + 1 )))
name="job_${n}.sh"
install -m 0644 "$job" "$BRIDGE/inbox/$name"
echo "queued $name -> $BRIDGE/inbox/"
[ "$wait" = 0 ] && exit 0
end=$(( $(date +%s) + timeout_s ))
while [ ! -f "$BRIDGE/outbox/$name.log" ]; do
  if [ $(date +%s) -gt $end ]; then
    echo "TIMEOUT after ${timeout_s}s waiting for $name.log (worker up? job still running?)" >&2
    exit 4
  fi
  sleep 15
done
echo "== $name.log =="; cat "$BRIDGE/outbox/$name.log"
# reap: an un-reaped job stays in inbox and the worker's log-exists check is the only
# thing keeping it from re-running, so take it out of the queue ourselves
mkdir -p "$BRIDGE/inbox_done"
if mv "$BRIDGE/inbox/$name" "$BRIDGE/inbox_done/" 2>/dev/null; then
  echo "(reaped $name -> $BRIDGE/inbox_done/)"
fi
