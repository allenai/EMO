#!/usr/bin/env bash
# Merge the four sub-models at matched progress point i (0..4 <-> baseline steps 20000/25000/30000/35000/38148),
# waiting for the four fixed-step checkpoints, then launch the merged model's evaluations on Beaker
# (held-out 20B-window CE unrestricted + oracle group routing; v3-small ppl validation).
#   bash scripts/sparse_experts/olmoe3_squares/merge_point.sh <i>
set -u
cd "$(git rev-parse --show-toplevel)"
i=$1; BASE=(20000 25000 30000 35000 38148); NAME=match${BASE[$i]}
SQN="${SQUARES_NAME:-olmoe3_squares}"; RP="${SQUARE_RUN_PREFIX:-olmoe3_275m_emo_square}"; FULL="${FULL_RUN:-olmoe3_275m_emo_10b}"; HR="${HELDOUT_DIR:-runs_heldout20b}"
# fixed checkpoint steps of the four squares (from their launch logs); override with STEPS_G0..3="a b c d e"
S0=(${STEPS_G0:-212 1354 2496 3638 4357}); S1=(${STEPS_G1:-370 2367 4364 6361 7618}); S2=(${STEPS_G2:-140 893 1647 2401 2875}); S3=(${STEPS_G3:-205 1314 2422 3530 4228})
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
subs=("$S/${RP}0/step${S0[$i]}" "$S/${RP}1/step${S1[$i]}" "$S/${RP}2/step${S2[$i]}" "$S/${RP}3/step${S3[$i]}")
for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 60; done; done
echo "$(date -u +%H:%M) all four checkpoints present for $NAME"
export PATH=/root/.conda/envs/emo/bin:$PATH
OUT=$S/$SQN/merged/$NAME
if [ ! -f "$OUT/merge_info.json" ]; then
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $S/$SQN/groups.json --full $S/$FULL/step19074 \
    --subs "$(IFS=,; echo "${subs[*]}")" --weights "${MERGE_WEIGHTS:-0.2284,0.3993,0.1507,0.2216}" --out "$OUT" --overwrite | tail -1 | cut -c1-160
fi
M="$W/$SQN/merged/$NAME"; R="$W/olmoe3_routing/$HR/merged_$NAME"
launch() { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$1" --gpus 1 --allocated -- "${@:2}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aoE "beaker.org/ex/[A-Z0-9]+" | head -1 | sed "s|^|$1: |"; }
launch "$SQN-eval-$NAME-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$M" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$R/none/rank0" --restrict none --batch-size 8 --log-every 100
launch "$SQN-eval-$NAME-oracle" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$M" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$R/oracle/rank0" --group-restrict "$W/$SQN/groups.json" --batch-size 8 --log-every 100
launch "$SQN-eval-$NAME-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$M" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
