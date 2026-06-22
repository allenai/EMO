# PARENT: "scripts/models_v2/stdmoe_1b14b_50bof130b.sh"
# DESCRIPTION:
#     - Standard top-k MoE (model type moe_lbreducedp_sharedexp) with 64 experts (vs 128
#       in the stdmoe_1b14b_50bof130b baseline), trained as a TRUE 50B-token run: the LR
#       cosine schedule decays directly over 50B (max_duration=50B, NO 130B/hard_stop
#       split), i.e. "what if we only ever had 50B tokens." 1 shared expert, lr 4e-3,
#       lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs.
#     - Permanent checkpoints at 25% / 50% / 75% of training via the checkpointer's
#       fixed_steps list. The 50B run is 11,921 steps at global_batch_size 1024 * seq
#       4096 = 4,194,304 tokens/step:
#         25% -> step 2980   (12.50B tokens)
#         50% -> step 5960   (25.00B tokens)
#         75% -> step 8941   (37.50B tokens)
#       A final ~50B checkpoint (step 11921) is also written at end-of-run, and 2-deep
#       rolling ephemerals are kept for preemption safety. Periodic permanent saves are
#       disabled (save_interval set huge) so ONLY the fixed_steps + final survive.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=4e-3
lb=1e-1

num_shared_experts=1
num_experts=64

runname="stdmoe_64exp_50b"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 50_000_000_000, unit: tokens}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[2980, 5960, 8941]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
