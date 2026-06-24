# PARENT: "scripts/models_v2/stdmoe_64exp_50b_wsd.sh"
# DESCRIPTION:
#     - A pure WSD STABLE TRUNK (standard top-k MoE, model type moe_lbreducedp_sharedexp,
#       64 experts, 50B-token run, 1 shared expert, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs).
#       Differs from stdmoe_64exp_50b_wsd.sh in three ways:
#         1. Peak LR is 2e-3 (vs 4e-3).
#         2. NO decay phase -- this run is a stable trunk only. We do NOT bake in an end-of-run
#            anneal; instead, decay branches are forked later from any saved checkpoint via
#            scripts/models_v2/launch_wsd_decay.sh (which auto-extracts the 2e-3 peak LR from the
#            branch checkpoint and decays to 0 over a chosen token budget). This lets us decide
#            WHEN and HOW LONG to anneal after the trunk has trained, branching cheaply at will.
#            Implemented as WSD with decay_steps=1: the LR is flat at the peak for the whole run
#            and only the single final optimizer step touches 0. (decay_steps=0 would divide by
#            zero in _linear_decay at the last step; 1 is the minimal crash-free "no decay".)
#         3. Permanent checkpoints every 5B tokens (not just 25/50/75%) so ANY 5B point is a
#            brancheable stable checkpoint -- the every-5B cadence the WSD decay-family convention
#            calls for (see launch_wsd_decay.sh header).
#     - The 50B run is 11,921 steps at global_batch_size 1024 * seq 4096 = 4,194,304 tokens/step.
#       fixed_steps = round(k * 5e9 / 4,194,304) for k=1..9 (5,10,...,45B):
#         5B -> 1192   10B -> 2384   15B -> 3576   20B -> 4768   25B -> 5960
#         30B -> 7153  35B -> 8345   40B -> 9537   45B -> 10729
#       The 50B final (step 11921) is auto-saved at end-of-run. Periodic permanent saves are
#       disabled (save_interval huge) so ONLY fixed_steps + final survive; 2-deep rolling
#       ephemerals are kept for preemption safety.
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
num_experts=64

warmup_steps=2000
decay_steps=1   # pure stable trunk: flat at peak LR; only the final step touches 0. No real anneal.

runname="stdmoe_64exp_50b_wsd_lr2e-3"

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
