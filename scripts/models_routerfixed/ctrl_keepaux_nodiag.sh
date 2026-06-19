# CONTROL for the graddiag runs (models_routerfixed): IDENTICAL to graddiag_keepaux.sh but with the
# per-group gradient diagnostic OFF (EMO_GRAD_DIAG unset). Purpose: disambiguate whether the diagnostic
# (which inserts extra .full_tensor()/all_reduce collectives into the step) was SUPPRESSING the
# frozen-router aux NaN, or whether the original probe NaNs (keepaux@120, lbonly@105, zonly@289) were
# horizon/stochastic. Both graddiag reruns (lbonly@150, keepaux@500) trained CLEAN past their original
# NaN steps; this run removes the only remaining difference. NaN here => diagnostic was causal; clean
# 500 steps here => the short-probe NaNs were stochastic and the "aux must be off" claim needs softening.
#
# keepaux recipe (lb=1e-1, z=1e-3), frozen grafted router, 8 nodes, eval off, no checkpoints, 500 steps.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_routerfixed"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

# Turn on the per-group gradient diagnostic inside the train module (no-op without this var).
# Local torchrun reads it from the env; Beaker workers get it via launch_common's BEAKER_ENV_VARS.

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1     # LB ON  (baseline)
zloss=1e-3  # Z  ON  (baseline) -- keepaux = both aux losses, the real recipe that NaN'd

num_shared_experts=1

runname="ctrl_routerfixed_keepaux_nodiag"
INIT_CHECKPOINT="${MODELS_DIR}/init_routerfixed_step0/model_and_optim"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--load_path="${INIT_CHECKPOINT}" \
		--load_trainer_state=false \
		--load_optim_state=false \
		--model.freeze_params='[blocks.*.feed_forward_moe.router.*]' \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 500, unit: steps}' \
		--trainer.callbacks.checkpointer.save_interval=100000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=99999 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[probe, ${EXPERIMENT_NAME}, routerfixed, keepaux, control-nodiag]" \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--model.block.feed_forward_moe.z_loss_weight=${zloss}
