# PARENT: "scripts/models_routerfixed/emo_1b14b_50bof130b_routerfixed_keepaux.sh"
# PHASE B grad-norm diagnostic (models_routerfixed) -- FAITHFUL capture run.
#
# The lbonly-only diagnostic run trained clean to 150 steps even though lbonly NaN'd as a probe at
# step 105: the frozen-router aux-loss NaN is a marginal bf16 instability, STOCHASTIC in timing
# (data order + cross-node NCCL bf16 rounding), not a deterministic step-105 divergence. To catch a
# divergence WITHOUT amplifying any weight, use the real keepaux recipe (BOTH aux losses on:
# lb=1e-1, z=1e-3) -- the production config that NaN'd at step 120 -- and stack both pressures to
# maximize the organic NaN probability, over a long horizon (500 steps).
#
# EMO_GRAD_DIAG=1 records per-bucket grad L2 norm + max-abs (router / ff_norm_gamma / attn_norm /
# experts / attention) under the `graddiag` WandB namespace, BEFORE clipping. The frozen `router`
# bucket has no grad (grad is None) so it is absent -- a built-in freeze sanity check.
#
# HYPOTHESIS (falsifiable): in the steps before the NaN, the `ff_norm_gamma`
# (feed_forward_norm.weight = the prenorm gain feeding the MoE) bucket ramps superlinearly and leads
# all non-expert buckets (max-abs grad overflows there first). If the ramp is uniform across buckets,
# or the experts lead, the "aux gradient dumps onto gamma" story is wrong.
#
# 8 nodes (mirror the runs that NaN'd), eval off, no checkpoints, capped at 500 steps.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_routerfixed"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

# Turn on the per-group gradient diagnostic inside the train module (no-op without this var).
# Local torchrun reads it from the env; Beaker workers get it via launch_common's BEAKER_ENV_VARS.
export EMO_GRAD_DIAG=1
BEAKER_ENV_VARS=(EMO_GRAD_DIAG=1)

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1     # LB ON  (baseline)
zloss=1e-3  # Z  ON  (baseline) -- keepaux = both aux losses, the real recipe that NaN'd

num_shared_experts=1

runname="graddiag_routerfixed_keepaux"
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
		--trainer.callbacks.wandb.tags="[probe, ${EXPERIMENT_NAME}, routerfixed, keepaux, graddiag]" \
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
