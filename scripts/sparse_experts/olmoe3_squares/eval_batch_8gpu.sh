#!/usr/bin/env bash
# Run a file of single-GPU evaluation commands inside ONE 8-GPU job, eight at a time, each pinned to its own GPU (2026-11-08: the allocated
# 1-GPU queue gave no slot in 4 h while whole-node grants came within an hour). Lines starting with # are skipped; a line's own output
# path is not checked here, so build the file from passes that are still missing.
#   bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh <command file>
set -u; F="${1:?command file}"; N=0; pids=(); gpu=0
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
say() { echo "== $(date -u +%m-%d\ %H:%M) [batch8] $*"; }
run_one() { local g=$1 cmd=$2; say "gpu $g: $cmd" | cut -c1-200; CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "/results/pass_$3.log" 2>&1 || say "gpu $g: FAILED ($3): $(tail -n 3 /results/pass_$3.log | tr '\n' ' ' | cut -c1-300)"; }
mkdir -p /results; i=0
while IFS= read -r line; do [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 20; done
  g=$(( i % 8 )); run_one $g "$line" $i & i=$((i+1)); sleep 5
done < "$F"
wait; say "all $i commands finished"
