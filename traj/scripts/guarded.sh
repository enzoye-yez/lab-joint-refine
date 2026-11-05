#!/usr/bin/env bash
# Run a heavy command in its own cgroup scope so it can never take the devbox down.
#
#   scripts/guarded.sh [--mem 400G] [--cpus 96] -- <command...>
#
# Kernel-enforced: memory cap (no swap), CPU quota, low priority, task limit. If the command
# exceeds the cap, the kernel kills only this scope; the shell, the editor and other jobs survive.
# Defaults: a quarter of the machine's memory and half of its CPUs.
set -euo pipefail
total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
mem="$(( total_kb / 4 / 1024 / 1024 ))G"
cpus=$(( $(getconf _NPROCESSORS_ONLN) / 2 ))  # not nproc: it honours OMP_NUM_THREADS
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem) mem="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
# One BLAS/OpenMP thread per process: multi-process jobs otherwise spawn cores x workers threads
# (64 OpenBLAS threads per worker drove the load to 1700+ on 192 cores).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec systemd-run --user --scope --quiet --collect \
  -p MemoryMax="$mem" -p MemorySwapMax=0 \
  -p CPUQuota="$(( cpus * 100 ))%" -p TasksMax=8192 \
  nice -n 10 "$@"
