#!/usr/bin/env bash
# Re-extract the held-out routing arrays (doc_usage etc.) of the merged passes of one control arm whose arrays were deleted by the
# storage policy (cleanup_heldout_arrays.py), for routing_drift.py (KL from the 10B start model; user request 2026-09-24).
# One allocated 1-GPU job per matched checkpoint, written to <pass>/none/rerun/ (the consolidated none/ folder keeps its counts and
# meta.json; nothing is deleted). Skips points already re-extracted. Usage: bash scripts/sparse_experts/olmoe3_squares/reextract_heldout_arrays.sh stdrand8
set -u
A="${1:?stdrand8|emorand8}"; SQN=olmoe3_squares_$A; HR=runs_heldout300b_$A; S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
STEPS="19074 20000 25000 30000 35000 38148 39073 44073 49073 54073 57221 66481 116479 166478 216477 247956"
say() { echo "$(date -u +%m-%d\ %H:%M) [reextract $A] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
for s in $STEPS; do
  d=$S/olmoe3_routing/$HR/merged_match$s/none
  [ -f $d/doc_usage.npy ] || [ -f $d/rank0/doc_usage.npy ] || [ -f $d/rerun/doc_usage.npy ] && { say "match$s: arrays present"; continue; }
  [ -d $S/$SQN/merged/match$s ] || { say "match$s: merged checkpoint missing, skipped"; continue; }
  launch "$SQN-reextract-match$s" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$SQN/merged/match$s" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/merged_match$s/none/rerun" --restrict none --batch-size 8 --log-every 100
done
