#!/usr/bin/env bash
# Stop a running scene batch: the batch runner and every descendant (scene worker, sandboxed
# codex sessions), deepest first, so the run lock is released. Requests cut off here keep an
# unknown outcome; resume with run_scene_batch.py --resume --execute --retry-uncertain.
set -euo pipefail
batch="${1:?usage: stop_batch.sh <batch name>}"
runner=$(ps -eo pid=,args= | awk -v b="--batch $batch" '$2 ~ /python/ && /run_scene_batch\.py/ && index($0, b" ") {print $1}')
[ -n "$runner" ] || { echo "no running batch $batch"; exit 0; }
descendants() { local p; for p in $(ps -o pid= --ppid "$1"); do descendants "$p"; echo "$p"; done; }
pids="$(descendants "$runner") $runner"
kill $pids 2>/dev/null || true
for _ in $(seq 30); do
  alive=$(for p in $pids; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
  [ -z "$alive" ] && { echo "stopped $batch: $pids"; exit 0; }
  sleep 1
done
kill -9 $alive 2>/dev/null || true
echo "stopped $batch (forced: $alive)"
