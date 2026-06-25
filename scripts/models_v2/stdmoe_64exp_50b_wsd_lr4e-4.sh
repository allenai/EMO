# PARENT: "scripts/models_v2/stdmoe_64exp_50b_wsd_lr2e-3.sh"
# DESCRIPTION:
#     - Identical to stdmoe_64exp_50b_wsd_lr2e-3.sh (pure WSD STABLE TRUNK, 64-expert stdMoE,
#       50B tokens, 1 shared expert, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs) EXCEPT the peak
#       LR is 4e-4 (vs 2e-3).
#     - NO decay phase: this is a stable trunk only (decay_steps=1 -> flat LR for the whole run,
#       only the single final optimizer step touches 0). Decay branches are forked later from any
#       saved checkpoint via scripts/models_v2/launch_wsd_decay.sh, which auto-extracts the 4e-4
#       peak LR from the branch checkpoint and decays to 0 over a chosen token budget.
#     - Permanent checkpoints every 5B tokens so any 5B point is a brancheable stable checkpoint.
#       The 50B run is 11,921 steps at global_batch_size 1024 * seq 4096 = 4,194,304 tokens/step:
#         5B->1192  10B->2384  15B->3576  20B->4768  25B->5960
#         30B->7153 35B->8345  40B->9537  45B->10729
#       The 50B final (step 11921) is auto-saved at end-of-run; periodic permanent saves are
#       disabled (save_interval huge) so ONLY fixed_steps + final survive; 2-deep rolling
#       ephemerals are kept for preemption safety.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=4e-4
lb=1e-1

num_shared_experts=1
num_experts=64

warmup_steps=2000
decay_steps=1   # pure stable trunk: flat at peak LR; only the final step touches 0. No real anneal.

runname="stdmoe_64exp_50b_wsd_lr4e-4"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 50_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[1192, 2384, 3576, 4768, 5960, 7153, 8345, 9537, 10729]" \
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
