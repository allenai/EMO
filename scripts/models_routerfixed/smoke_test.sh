#!/usr/bin/env bash
# 1-GPU smoke test for the models_routerfixed pipeline (shrunk model so it fits/iterates fast).
#
# Validates the full mechanism end-to-end before the expensive 8-node launch:
#   [1] tiny pretrain (no freeze) -> a source checkpoint with non-trivial "trained" routers
#   [2] build_step0_routerfixed.py -> graft those routers onto a fresh init (model-only checkpoint)
#   [3] tiny pretrain WITH frozen routers, loading the grafted init at step 0 (fresh optimizer),
#       saving checkpoints every few steps
#   [4] check_frozen.py -> assert routers are bit-identical across the saved steps (freeze works),
#       other weights changed (training works), and routers still equal the grafted init.
#
#   bash scripts/models_routerfixed/smoke_test.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$(pwd)/src"

OUT="claude_outputs/models_routerfixed/smoke"
WORK_DIR="${DATASET_CACHE:-$HOME/dataset-cache}"
DATA_ROOT="${DATA_ROOT:-s3://ai2-llm}"
rm -rf "$OUT"; mkdir -p "$OUT"

# Shrunk model (n_layers=2, d_model=128, 8 experts, hidden=64) -- one router.weight per layer.
#
# NOTE on router/backend choice: this local session has no flash-attn, and the production randpool
# router needs intra-document masking (only the flash backends support it). Freezing, however, is
# router-class-agnostic -- freeze_params just sets requires_grad=False on `*.router.weight`, and
# build_step0 only copies that tensor -- so the smoke uses the simple `moe` router with the torch
# backend (no doc masking) to exercise the exact freeze + load + checkpoint mechanism. (That the
# production randpool router's params freeze correctly is checked separately/statically.)
TINY=(
  --model-type=moe
  --model.block.feed_forward_moe.num_experts=8
  --model.d_model=128 --model.n_layers=2
  --model.block.sequence_mixer.n_heads=4
  --model.block.sequence_mixer.backend=torch
  --model.block.feed_forward_moe.hidden_size=64
  --model.block.sequence_mixer.qk_norm=null
  --model.block.name=moe
  --dataset.mix=OLMoE-mix-0824
  --global_batch_size=4
  --lr=4e-3
)
COMMON=(
  --work-dir="$WORK_DIR" --data-root="$DATA_ROOT"
  --trainer.callbacks.wandb.enabled=false
  --trainer.callbacks.downstream_evaluator.enabled=false
  --trainer.callbacks.checkpointer.save_interval=1000
  --trainer.callbacks.checkpointer.ephemeral_save_interval=5
  --trainer.callbacks.checkpointer.keep_ephemeral=5
  --trainer.callbacks.checkpointer.pre_train_checkpoint=false
  "--trainer.max_duration={value: 12, unit: steps}"
)

echo "### [1/4] tiny pretrain (no freeze) -> source checkpoint with trained routers"
torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py smoke_src \
  --save-folder="$OUT/src" "${TINY[@]}" "${COMMON[@]}"
SRC=$(ls -d "$OUT"/src/step* | sort -V | tail -1)
echo "    source checkpoint: $SRC"

echo "### [2/4] build the router-fixed init from that source"
python scripts/models_routerfixed/build_step0_routerfixed.py \
  --src-checkpoint "$SRC" --out-dir "$OUT/init"

echo "### [3/4] tiny pretrain WITH frozen routers, loading the grafted init at step 0"
torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py smoke_frz \
  --save-folder="$OUT/frz" "${TINY[@]}" "${COMMON[@]}" \
  --load_path="$OUT/init/model_and_optim" \
  --load_trainer_state=false --load_optim_state=false \
  --model.freeze_params='[blocks.*.feed_forward_moe.router.*]'

echo "### [4/4] verify freeze + training + graft preservation"
A=$(ls -d "$OUT"/frz/step* | sort -V | head -1)
B=$(ls -d "$OUT"/frz/step* | sort -V | tail -1)
echo "    comparing $A vs $B"
python scripts/models_routerfixed/check_frozen.py --a "$A" --b "$B" --expect-router-equals "$OUT/init"

echo "SMOKE TEST PASSED"
