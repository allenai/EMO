# PARENT: "scripts/models_v2/stdmoe_128exp_50b.sh" (the lr4e-3 cosine twin)
# DESCRIPTION:
#     - Standard top-k MoE (model type moe_lbreducedp_sharedexp) with 128 experts, trained as a
#       TRUE 50B-token run with the default COSINE LR schedule decaying fully over 50B (no
#       hard_stop). 1 shared expert, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs.
#     - Identical to stdmoe_128exp_50b.sh EXCEPT the peak LR is 2e-3 (vs that run's 4e-3). This
#       exists to make the LR-scheduling comparison FAIR at a fixed peak LR: we already have
#       cosine@4e-3 vs WSD@4e-3, but the favored WSD trunk is at 2e-3 (stdmoe_128exp_50b_wsd_lr2e-3)
#       with no cosine counterpart -- so the scheduler effect at 2e-3 was confounded with the LR
#       change. This run is the cosine@2e-3 partner for that pairing.
#     - Permanent checkpoints at 25% / 50% / 75% (fixed_steps) + a final ~50B (step 11921);
#       2-deep rolling ephemerals for preemption safety; periodic permanent saves disabled.
#       11,921 steps at global_batch_size 1024 * seq 4096 = 4,194,304 tokens/step:
#         25% -> step 2980 (12.5B)   50% -> step 5960 (25B)   75% -> step 8941 (37.5B)
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/stdmoe_128exp_50b_cos_lr2e-3.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=128

runname="stdmoe_128exp_50b_cos_lr2e-3"

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
