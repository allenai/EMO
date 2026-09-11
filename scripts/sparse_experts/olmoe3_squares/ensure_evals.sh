#!/usr/bin/env bash
# Relaunch any missing evaluation of the olmoe3_squares (default) or olmoe3_squares_std (SQUARES_NAME=...) variant:
# for every matched point with an existing merged model / baseline checkpoint, check the held-out (none, oracle)
# outputs and the ppl json, and launch what is absent. Run when all training is done (jobs still running
# would be duplicated). Prints one line per launch.
cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
SQN="${SQUARES_NAME:-olmoe3_squares}"; HR="${HELDOUT_DIR:-runs_heldout20b}"; BRUN="${BASELINE_RUN:-olmoe3_275m_emo_20b_1node}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
launch() { local name=$1; shift; local log=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad/launch_$name.log; local u=""
  for attempt in 1 2 3 4; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 30   # gantry's git check fails with a broken pipe now and then
  done
  echo "$name: ${u:-LAUNCH FAILED after 4 attempts}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
for step in 20000 25000 30000 35000 38148; do
  # merged
  M=$S/$SQN/merged/match$step
  if [ -f "$M/merge_info.json" ]; then
    R=$S/olmoe3_routing/$HR/merged_match$step
    have $R/none || launch "$SQN-eval-match$step-none-r" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$SQN/merged/match$step" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/merged_match$step/none/rank0" --restrict none --batch-size 8 --log-every 100
    have $R/oracle || launch "$SQN-eval-match$step-oracle-r" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$SQN/merged/match$step" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/merged_match$step/oracle/rank0" --group-restrict "$W/$SQN/groups.json" --batch-size 8 --log-every 100
    [ -f $S/$SQN/ppl_validation/merged/match$step.json ] || launch "$SQN-eval-match$step-ppl-r" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$SQN/merged/match$step" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
  fi
  # baseline (final checkpoint may be step38147)
  for st in $step $((step-1)); do [ -d "$S/$BRUN/step$st/model_and_optim" ] && break; done
  if [ -d "$S/$BRUN/step$st/model_and_optim" ]; then
    R=$S/olmoe3_routing/$HR/baseline_step$step
    have $R/none || launch "$SQN-eval-base$step-none-r" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$BRUN/step$st" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/baseline_step$step/none/rank0" --restrict none --batch-size 8 --log-every 100
    have $R/oracle || launch "$SQN-eval-base$step-oracle-r" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$BRUN/step$st" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/baseline_step$step/oracle/rank0" --group-restrict "$W/$SQN/groups.json" --batch-size 8 --log-every 100
    [ -f $S/$SQN/ppl_validation/$BRUN/step$st.json ] || launch "$SQN-eval-base$step-ppl-r" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$BRUN/step$st" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
  fi
done
echo "ensure_evals done"
