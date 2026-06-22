# PARENT: "scripts/models_v2/stdmoe_64exp_50b.sh"
# DESCRIPTION:
#     - Standard top-k MoE (model type moe_lbreducedp_sharedexp) with 64 experts, trained
#       as a TRUE 25B-token run: the LR cosine schedule decays directly over 25B
#       (max_duration=25B, no hard_stop), i.e. "what if we only ever had 25B tokens."
#       1 shared expert, lr 4e-3, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs.
#       Sibling of stdmoe_64exp_50b.sh (identical except the token budget / LR horizon).
#     - Permanent checkpoints at 25% / 50% / 75% of training via the checkpointer's
#       fixed_steps list. The 25B run is 5,961 steps at global_batch_size 1024 * seq
#       4096 = 4,194,304 tokens/step:
#         25% -> step 1490   (6.25B tokens)
#         50% -> step 2980   (12.50B tokens)
#         75% -> step 4471   (18.75B tokens)
#       A final ~25B checkpoint (step 5961) is also written at end-of-run, and 2-deep
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

runname="stdmoe_64exp_25b"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 25_000_000_000, unit: tokens}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[1490, 2980, 4471]" \
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
