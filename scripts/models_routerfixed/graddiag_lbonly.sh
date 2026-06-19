# PARENT: "scripts/models_routerfixed/probe_aux_lbonly.sh"
# PHASE B grad-norm diagnostic (models_routerfixed): reproduce the frozen-router NaN with
# per-parameter-group gradient logging ON, to localize WHERE the blow-up originates.
#
# Re-runs the lbonly config (lb=1e-1, z=0) -- the fastest, deterministic NaN (step 105 in the
# 8-node probe) -- with EMO_GRAD_DIAG=1, which records per-bucket grad L2 norm and max-abs grad
# under the `graddiag` WandB namespace (router / ff_norm_gamma / attn_norm / experts / attention).
#
# HYPOTHESIS (falsifiable): in the ~10 steps before the NaN, the `ff_norm_gamma`
# (feed_forward_norm.weight = the prenorm gain feeding the MoE) bucket ramps superlinearly and
# leads all non-expert buckets, while `router` stays ~0 (frozen sanity). If the ramp is uniform
# across buckets, or the experts lead, the "aux gradient dumps onto gamma" story is wrong.
#
# 8 nodes (mirror the probe that NaN'd), eval off, no checkpoints, capped at 150 steps (dies ~105).
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
lb=1e-1   # LB ON (the fast-NaN aux loss)

num_shared_experts=1

runname="graddiag_routerfixed_lbonly"
INIT_CHECKPOINT="${MODELS_DIR}/init_routerfixed_step0/model_and_optim"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--load_path="${INIT_CHECKPOINT}" \
		--load_trainer_state=false \
		--load_optim_state=false \
		--model.freeze_params='[blocks.*.feed_forward_moe.router.*]' \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 150, unit: steps}' \
		--trainer.callbacks.checkpointer.save_interval=100000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=99999 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[probe, ${EXPERIMENT_NAME}, routerfixed, lbonly, graddiag]" \
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
