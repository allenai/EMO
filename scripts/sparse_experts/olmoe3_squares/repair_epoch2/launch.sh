#!/usr/bin/env bash
# Launch the epoch-2 repair of every square run: one allocated 8-GPU Beaker job per variant running worker.sh
# sequentially over the variant's squares (28 resumes of 24-375 steps). Renames each tainted final step<s+1> to
# step<s+1>_epoch2 first (the trainer then resumes from the latest remaining checkpoint). Commit + push before running.
#   bash scripts/sparse_experts/olmoe3_squares/repair_epoch2/launch.sh [variant ...]   (default: all six)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
source scripts/sparse_experts/olmoe3_squares/repair_epoch2/variants.sh
S=sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
VARIANTS="${*:-emo noemo std k8 pool64or512 learnedd}"
for V in $VARIANTS; do
  variant_env $V || exit 1; LOG=$S/$SQN/logs; mkdir -p $LOG
  [ -f $LOG/repair_launched ] && { echo "$V: already launched ($(cat $LOG/repair_launched))"; continue; }
  specs=($(variant_specs $SQN $RP $PACK $K)) || exit 1
  for spec in "${specs[@]}"; do IFS=: read -r g tokens s old resume <<< "$spec"; run=$S/${RP}$g
    [ -d $run/step$old ] && [ ! -d $run/step${old}_epoch2 ] && mv $run/step$old $run/step${old}_epoch2
    echo "$V square $g: $tokens tokens -> $s steps (was $old); resume from step$resume"
  done
  u=""; for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "repair-epoch2-$V" --gpus 8 --allocated -- bash scripts/sparse_experts/olmoe3_squares/repair_epoch2/worker.sh $V "${specs[@]}" > $SP/launch_repair_$V.log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/launch_repair_$V.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  echo "$(date -u +%H:%M) $V: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/repair_launched
done
