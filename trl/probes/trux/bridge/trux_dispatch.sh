#!/usr/bin/env bash
# trux side: enqueue a job for the Spark worker, optionally wait for its log.
#   trux_dispatch.sh job.sh [--wait] [--timeout 5400] [--hold]
#
# --hold stages the job in $BRIDGE/hold/ instead of the inbox. Until the worker moves a job into
# inbox_done it is still trux's to take back, so a job can be queued and then withdrawn - which is
# how a pass-7 lane was stopped after the audit showed 1.1 GiB free on the Spark box (a CPU lane of
# that size would have taken most of a day). Release it later by dispatching the held path.
# endpoints (bridge dir, hosts, keys) live in an untracked sibling file; see
# bridge.env.example. Public tree carries no lab hostnames or scratch paths.
_BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$_BRIDGE_DIR/bridge.env" ] && . "$_BRIDGE_DIR/bridge.env"
set -u
BRIDGE=${BRIDGE:?BRIDGE unset - copy bridge.env.example to bridge.env}
# flags first or last, both work: `--hold job.sh` is how the operator types it, and a job
# named --hold is not a thing worth supporting
job=""; wait=0; timeout_s=5400; hold=0
while [ $# -gt 0 ]; do
  case $1 in
    --wait) wait=1 ;;
    --timeout) timeout_s=$2; shift ;;
    --hold) hold=1 ;;
    -*) echo "FATAL: unknown flag $1" >&2; exit 2 ;;
    *) [ -z "$job" ] && job=$1 || { echo "FATAL: more than one job given ($job, $1)" >&2; exit 2; } ;;
  esac
  shift
done
[ -n "$job" ] || { echo "FATAL: no job script given (usage: trux_dispatch.sh job.sh [--wait] [--timeout SECS] [--hold])" >&2; exit 2; }
[ -f "$BRIDGE/outbox/heartbeat" ] || { echo "no heartbeat yet: the worker has not dialed in (or is dead)" >&2; }
if [ -f "$BRIDGE/outbox/heartbeat" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$BRIDGE/outbox/heartbeat") ))
  echo "heartbeat ${age}s old:"; sed 's/^/  /' "$BRIDGE/outbox/heartbeat"
  [ "$age" -gt 900 ] && echo "WARNING: worker silent for ${age}s — results may be stale" >&2
fi
if [ "$hold" = 1 ]; then
  # staged, not queued: no number is allocated, because the job may never be released
  mkdir -p "$BRIDGE/hold"
  install -m 0644 "$job" "$BRIDGE/hold/$(basename "$job")"
  echo "held $job -> $BRIDGE/hold/ (release: trux_dispatch.sh $BRIDGE/hold/$(basename "$job"))"
  exit 0
fi
# Number from the highest number ever seen, not from how many are finished. The worker only
# runs an inbox job that has no outbox log of the same name, so reusing a number is not a
# cosmetic problem: the job is silently never executed (it looked queued, the worker was
# alive, and there was no error anywhere). Counting *.done.sh misses jobs still in the inbox
# and jobs whose log exists while the file was already reaped.
last=$(ls -1 "$BRIDGE/inbox"/job_*.sh "$BRIDGE/inbox_done"/job_*.sh \
          "$BRIDGE/outbox"/job_*.log 2>/dev/null | sed -E 's/.*job_0*([0-9]+).*/\1/' | sort -n | tail -1)
n=$(printf '%03d' $((${last:-0} + 1)))
name="job_${n}.sh"
if [ -f "$BRIDGE/outbox/$name.log" ]; then
  echo "FATAL: $name.log already exists - the worker would skip this job" >&2; exit 5
fi
# Queueing a file that was generated in an earlier attempt is how job_021 ended up running the
# pre-fix template: the generator had aborted on a malformed bridge.env under set -e, its previous
# output was still sitting in /tmp, and dispatch happily shipped it. A generated job carries a
# stamp line; if it is missing or stale, what is being queued is not what the template says.
if ! head -2 "$job" | grep -q "^# GENERATED "; then
  echo "FATAL: $job carries no GENERATED stamp - run gpu7_generate.sh, do not queue a hand-kept copy" >&2
  exit 7
fi
gen_age=$(( $(date +%s) - $(sed -n '2s/^# GENERATED \([^ ]*\).*/\1/p' "$job" | xargs -I{} date -d {} +%s) ))
if [ "$gen_age" -gt 1800 ]; then
  echo "FATAL: $job was generated ${gen_age}s ago; regenerate it - the template may have changed since" >&2
  exit 7
fi
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
