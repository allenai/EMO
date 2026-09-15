#!/usr/bin/env bash
# Held-out passes (20B-window sample, unrestricted) of every square at the matched progress points, for the piecewise
# diagnostic; then merge + piecewise_eval per point. Launches one 1-GPU allocated Beaker job per (square, point).
#   POINTS="2 3 4" bash scripts/sparse_experts/olmoe3_squares/sub_passes.sh emo|std
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
V=$1; POINTS="${POINTS:-2 3 4}"; BASE=(20000 25000 30000 35000 38148)
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; S=sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
case $V in
  emo) HR=runs_heldout20b; RP=olmoe3_275m_emo_square; SQN=olmoe3_squares; START=emo_step19074
       ST=("212 1354 2496 3638 4357" "370 2367 4364 6361 7618" "140 893 1647 2401 2875" "205 1314 2422 3530 4228");;
  std) HR=runs_heldout20b_std; RP=olmoe3_275m_square; SQN=olmoe3_squares_std; START=std_step19074
       ST=("180 1149 2118 3088 3698" "251 1606 2961 4317 5170" "275 1757 3240 4723 5656" "221 1415 2609 3802 4554");;
  pool64or512|learnedd)
       HR=runs_heldout20b_$V; RP=olmoe3_275m_${V}_square; SQN=olmoe3_squares_$V; START=${V}_step19074
       ST=(); for g in 0 1 2 3; do ST+=("$(python -c "
import json,math; t=json.load(open('sparse_experts/$SQN/pack/stats.json'))['tokens_per_group'][$g]; s=math.ceil(t/524288)
print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')")"); done;;
  *) echo "emo|std|pool64or512|learnedd"; exit 1;;
esac
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 30; done
  echo "$(date -u +%H:%M) $name: ${u:-LAUNCH FAILED after 4 attempts}"; }
for i in $POINTS; do NAME=match${BASE[$i]}
  for g in 0 1 2 3; do steps=(${ST[$g]}); ck="$W/$RP$g/step${steps[$i]}"; R="$S/olmoe3_routing/$HR/sub${g}_$NAME/none"
    [ -f "$R/meta.json" ] || [ -f "$R/rank0/DONE" ] || launch "$SQN-sub$g-$NAME" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ck" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/sub${g}_$NAME/none/rank0" --restrict none --batch-size 8 --log-every 100
  done
done
for i in $POINTS; do NAME=match${BASE[$i]}
  for g in 0 1 2 3; do R="$S/olmoe3_routing/$HR/sub${g}_$NAME/none"
    until [ -f "$R/meta.json" ] || [ -f "$R/rank0/DONE" ]; do sleep 120; done
    [ -f "$R/meta.json" ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py "$S/olmoe3_routing/$HR/sub${g}_$NAME/none" 2>&1 | tail -1
  done
  HELDOUT_DIR=$HR SQUARES_NAME=$SQN START_TAG=$START python scripts/sparse_experts/olmoe3_squares/piecewise_eval.py --name $NAME | head -3
  echo "$(date -u +%H:%M) piecewise $V $NAME done"
done
