#!/usr/bin/env bash
# Wait for the continued-baseline checkpoint at matched point i (steps 20000/25000/30000/35000/38148, from
# whichever baseline copy is running: olmoe3_275m_emo_20b or olmoe3_275m_emo_20b_filler) and launch its
# evaluations (held-out 20B-window CE unrestricted + oracle group routing; v3-small ppl validation).
#   bash scripts/sparse_experts/olmoe3_squares/baseline_point.sh <i>
set -u
cd "$(git rev-parse --show-toplevel)"
i=$1; BASE=(20000 25000 30000 35000 38148); step=${BASE[$i]}
SQN="${SQUARES_NAME:-olmoe3_squares}"; HR="${HELDOUT_DIR:-runs_heldout20b}"; RUNS="${BASELINE_RUNS:-olmoe3_275m_emo_20b olmoe3_275m_emo_20b_filler olmoe3_275m_emo_20b_1node}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
while true; do
  for run in $RUNS; do for st in $step $((step-1)); do
    [ -f "$S/$run/step$st/train/rank0.pt" ] && { RUN=$run; STEP=$st; break 3; }
  done; done
  sleep 120
done
echo "$(date -u +%H:%M) baseline $RUN step$STEP present"
export PATH=/root/.conda/envs/emo/bin:$PATH
M="$W/$RUN/step$STEP"; R="$W/olmoe3_routing/$HR/baseline_step$step"
launch() { local name=$1; shift; local log=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad/launch_$name.log; local u=""
  for attempt in 1 2 3 4; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 30   # gantry's git check fails with a broken pipe now and then
  done
  echo "$name: ${u:-LAUNCH FAILED after 4 attempts}"; }
launch "$SQN-eval-base$step-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$M" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$R/none/rank0" --restrict none --batch-size 8 --log-every 100
launch "$SQN-eval-base$step-oracle" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$M" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$R/oracle/rank0" --group-restrict "$W/$SQN/groups.json" --batch-size 8 --log-every 100
launch "$SQN-eval-base$step-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$M" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
