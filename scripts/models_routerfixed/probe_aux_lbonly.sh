# PARENT: "scripts/models_routerfixed/emo_1b14b_130b_routerfixed_keepaux.sh"
# DIAGNOSTIC PROBE (models_routerfixed): isolate which auxiliary loss destabilizes the frozen-router
# run. keepaux (lb=1e-1 AND z=1e-3) NaN'd at step ~119 -- the NaN was in the GRADIENT (grad norm went
# NaN while every loss term stayed finite), pointing at an aux-loss backward overflowing bf16. Theory:
# the router z-loss gradient onto the hidden state scales ~||W||^2 (vs ||W|| for the LB loss), and the
# grafted trained router has ||W||~59-69 (vs ~10 fresh), so z-loss is the prime suspect.
#
# THIS PROBE = LB ONLY (lb=1e-1, z=0). Prediction: roughly STABLE past step ~119.
# Sibling probe_aux_zonly.sh = Z ONLY (lb=0, z=1e-3) -- predicted to NaN.
#
# Short + cheap: 1 node, ~400 steps, eval off, no checkpoints. Loads the same grafted init + frozen
# router as keepaux. (Node count is not part of the hypothesis -- the ||W||^2 overflow is per-token in
# the backward -- so 1 node is used for fast scheduling; the exact NaN step may differ from the 8-node
# keepaux run, but a systematic z-loss overflow should still appear within a few hundred steps.)
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_routerfixed"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=1
BEAKER_GPUS=8

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1   # LB ON

num_shared_experts=1

runname="probe_routerfixed_lbonly"
INIT_CHECKPOINT="${MODELS_DIR}/init_routerfixed_step0/model_and_optim"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--load_path="${INIT_CHECKPOINT}" \
		--load_trainer_state=false \
		--load_optim_state=false \
		--model.freeze_params='[blocks.*.feed_forward_moe.router.*]' \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 400, unit: steps}' \
		--trainer.callbacks.checkpointer.save_interval=100000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=99999 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[probe, ${EXPERIMENT_NAME}, routerfixed, lbonly]" \
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
		--model.block.feed_forward_moe.z_loss_weight=0
