# PARENT: "scripts/models_v2/emo_64exp_50b_wsd_lr2e-3.sh"
# DESCRIPTION:
#     Local GPU smoke test for the meta_learning train module on a TINY model (2 layers, 16
#     experts, d_model default) with real data. Runs every meta_mode for ~30 steps with
#     EMO_META_CHECK_RESTORE=1 (bitwise weight-restore assert every step), then an inner-lr
#     mini-sweep in same_tokens mode to pick the pilot alpha by the
#     `train/meta delta/weight norm` metric (target ~1e-3..1e-2) and outer-CE stability.
#
#     Oracle A (vanilla-mode bit-identity with the parent trainer) is covered because
#     meta_mode=vanilla delegates to super().train_batch(); set RUN_PARENT_REF=1 to also run the
#     parent entry script with matched flags and compare console CE curves by eye.
#
#     Run scripts/meta_learning/model_scripts/verify_meta_step.py FIRST (the mechanism-correctness gate).
#
#   bash scripts/meta_learning/model_scripts/smoke_local.sh                # all modes + alpha sweep
#   MODES="same_tokens" ALPHAS="3e-2" bash scripts/meta_learning/model_scripts/smoke_local.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

EXPERIMENT_NAME="meta_learning"
# In a GPU-attached session the weka save root corresponds to the repo tree; keep smoke outputs
# under ./meta_learning/smoke/. NOTE: launch_common.sh already assigned MODELS_DIR at source
# time, so the assignment must be UNCONDITIONAL (a `${MODELS_DIR:-...}` here would be a no-op).
MODELS_DIR="$(pwd)/meta_learning"
DATA_ROOT="s3://ai2-llm"
DATASET_CACHE="$HOME/dataset-cache"  # unconditional: launch_common already set the weka default

export EMO_META_CHECK_RESTORE=1

MODE=local
NPROC="${NPROC:-2}"

STEPS="${STEPS:-30}"
MODES="${MODES:-vanilla outer_only same_tokens heldout}"
ALPHAS="${ALPHAS:-3e-3 3e-2 3e-1}"

run_smoke() {
    local runname="$1"; shift
    launch src/scripts/train/olmoe-1B-7B_fsl_meta.py "$runname" \
        --save-folder="${MODELS_DIR}/smoke/${runname}" \
        --dataset.mix=OLMoE-mix-0824 \
        --work-dir="${DATASET_CACHE}" \
        --trainer.hard_stop="{value: ${STEPS}, unit: steps}" \
        --trainer.max_duration='{value: 10_000_000_000, unit: tokens}' \
        --scheduler=wsd \
        --warmup_steps=10 \
        --decay_steps=1 \
        --global_batch_size=16 \
        --trainer.no_checkpoints=true \
        --trainer.callbacks.checkpointer.save_interval=1000000 \
        --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
        --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
        --trainer.callbacks.wandb.enabled=false \
        --trainer.callbacks.downstream_evaluator.enabled=false \
        --trainer.callbacks.lm_eval_full.enabled=false \
        --trainer.callbacks.lm_eval_pool32.enabled=false \
        --model.n_layers=2 \
        --model.block.feed_forward_moe.num_experts=16 \
        --dataset.generate_doc_lengths=false \
        --min_document_expert_pool=4 \
        --max_document_expert_pool=16 \
        --eval_document_expert_pool=16 \
        --num_shared_experts=1 \
        --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
        --model.block.name="moe" \
        --model.block.sequence_mixer.qk_norm=null \
        --lr=2e-3 \
        --model.block.feed_forward_moe.lb_loss_weight=1e-1 \
        --train_module.compile_model=false \
        --train_module.inner_pool_size=4 \
        "$@"
}

for mode in ${MODES}; do
    echo "=== smoke: meta_mode=${mode} ==="
    run_smoke "smoke_${mode}" \
        --train_module.meta_mode="${mode}" \
        --train_module.inner_lr=3e-2
done

echo "=== smoke: alpha sweep (same_tokens) — pick alpha with delta/weight ~1e-3..1e-2 ==="
for alpha in ${ALPHAS}; do
    echo "--- alpha=${alpha} ---"
    run_smoke "smoke_alpha_${alpha}" \
        --train_module.meta_mode=same_tokens \
        --train_module.inner_lr="${alpha}"
done

if [[ "${RUN_PARENT_REF:-0}" == "1" ]]; then
    echo "=== parent-trainer reference (compare CE curve vs smoke_vanilla) ==="
    launch src/scripts/train/olmoe-1B-7B_fsl.py "smoke_parent_ref" \
        --save-folder="${MODELS_DIR}/smoke/smoke_parent_ref" \
        --dataset.mix=OLMoE-mix-0824 \
        --work-dir="${DATASET_CACHE}" \
        --trainer.hard_stop="{value: ${STEPS}, unit: steps}" \
        --trainer.max_duration='{value: 10_000_000_000, unit: tokens}' \
        --scheduler=wsd \
        --warmup_steps=10 \
        --decay_steps=1 \
        --global_batch_size=16 \
        --trainer.no_checkpoints=true \
        --trainer.callbacks.checkpointer.save_interval=1000000 \
        --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
        --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
        --trainer.callbacks.wandb.enabled=false \
        --trainer.callbacks.downstream_evaluator.enabled=false \
        --model.n_layers=2 \
        --model.block.feed_forward_moe.num_experts=16 \
        --dataset.generate_doc_lengths=false \
        --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
        --min_document_expert_pool=4 \
        --max_document_expert_pool=16 \
        --eval_document_expert_pool=16 \
        --num_shared_experts=1 \
        --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
        --model.block.name="moe" \
        --model.block.sequence_mixer.qk_norm=null \
        --lr=2e-3 \
        --model.block.feed_forward_moe.lb_loss_weight=1e-1 \
        --train_module.compile_model=false
fi
