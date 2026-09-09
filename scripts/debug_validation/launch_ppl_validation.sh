#!/usr/bin/env bash
# Offline v3-small ppl validation of the two OLMoE3-ladder 275M 512-expert models (standard vs EMO),
# every permanent checkpoint (steps 5000/10000/15000/19000/19074), on Beaker: one ALLOCATED 1-GPU
# jupiter job per model that walks its checkpoints sequentially (~8 min each on an H100), writing
#   claude_outputs/debug_validation/ppl_validation/<run>/step<N>.json   (= weka; skips existing)
# Uses the pinned submodule stack via scripts/sparse_experts/olmoe3_beaker_cmd.py. Push first.
#
#   bash scripts/debug_validation/launch_ppl_validation.sh                 # both models
#   RUNS="olmoe3_275m_emo_10b" bash scripts/debug_validation/launch_ppl_validation.sh
#   STEPS="step19074" ... to restrict checkpoints; DRY=1 to print the launch config only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PATH=/root/.conda/envs/emo/bin:$PATH
unset GH_TOKEN   # the session's GH_TOKEN is invalid ("invalid header field value") -> gantry falls back to the gh CLI
LOG=${LOG:-/tmp/claude-0/pplval_launch}; mkdir -p "$LOG"
W=/weka/oe-training-default/ryanwang/EMO
RUNS=${RUNS:-"olmoe3_275m_10b olmoe3_275m_emo_10b"}
STEPS=${STEPS:-"step5000 step10000 step15000 step19000 step19074"}
OUT=${OUT:-$W/claude_outputs/debug_validation/ppl_validation}
for run in $RUNS; do
  ckpts=(); for s in $STEPS; do ckpts+=("$W/sparse_experts/$run/$s"); done
  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py \
    --name "pplval-$run" --gpus 1 --allocated ${DRY:+--dry-run} -- \
    python scripts/debug_validation/eval_ppl_validation.py --checkpoints "${ckpts[@]}" --out-dir "$OUT" --batch-size 8 \
    2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$LOG/$run.log" | grep -oE 'beaker.org/ex/[A-Z0-9]+|BeakerLaunchConfig.*|Error.*|error.*' | head -1 | sed "s|^|$run: |"
done
