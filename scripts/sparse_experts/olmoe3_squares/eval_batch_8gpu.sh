#!/usr/bin/env bash
# Run a file of single-GPU evaluation commands inside ONE 8-GPU job, eight at a time, each on a GPU that is actually free (2026-11-08: the
# allocated 1-GPU queue gave no slot in 10 h while whole-node grants came within hours; v1 assigned GPUs round-robin and two 2000e passes
# collided on one GPU -> OOM). Lines starting with # are skipped; a line's own output path is not checked here, so build the file from
# passes that are still missing. Per-command logs: /results/pass_<i>.log.
#   bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh <command file>
set -u; F="${1:?command file}"
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
say() { echo "== $(date -u +%m-%d\ %H:%M) [batch8] $*"; }
mkdir -p /results; declare -a PID=(0 0 0 0 0 0 0 0); i=0; fails=0
free_gpu() { local g; while true; do for g in 0 1 2 3 4 5 6 7; do [ "${PID[$g]}" = 0 ] && { echo $g; return; }; kill -0 "${PID[$g]}" 2>/dev/null || { wait "${PID[$g]}" 2>/dev/null || fails=$((fails+1)); PID[$g]=0; echo $g; return; }; done; sleep 15; done; }
while IFS= read -r line; do [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
  g=$(free_gpu); say "gpu $g <- command $i: $(echo "$line" | cut -c1-160)"
  ( CUDA_VISIBLE_DEVICES=$g bash -c "$line" > "/results/pass_$i.log" 2>&1 || { say "command $i FAILED on gpu $g: $(tail -n 2 /results/pass_$i.log | tr '\n' ' ' | cut -c1-240)"; exit 1; } ) & PID[$g]=$!; i=$((i+1)); sleep 5
done < "$F"
for g in 0 1 2 3 4 5 6 7; do [ "${PID[$g]}" = 0 ] || wait "${PID[$g]}" || fails=$((fails+1)); done
say "all $i commands finished, $fails failed"; [ $fails = 0 ]
