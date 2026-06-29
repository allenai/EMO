# PARENT: "scripts/models_v2/stdmoe_64exp_50b_wsd_lr2e-3.sh"
# DESCRIPTION:
#     - A pure WSD STABLE TRUNK (standard top-k MoE, model type moe_lbreducedp_sharedexp,
#       32 experts, 50B-token run, 1 shared expert, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs).
#       Identical to stdmoe_64exp_50b_wsd_lr2e-3.sh EXCEPT num_experts=32 (vs 64). top_k stays 8
#       (inherited from the base config, NOT scaled by num_experts), so active compute is unchanged
#       (8 routed + 1 shared); only TOTAL capacity shrinks -> a lower-capacity baseline that sits
#       further below 128e. Purpose: widen the lower->upper gap for the extension experiments
#       (64e->128e was a small gap; a 32e/16e source makes extension's benefit more visible).
#     - Peak LR 2e-3, flat WSD stable trunk (decay_steps=1: flat at peak, only the final step
#       touches 0). Decay branches are forked later via launch_wsd_decay.sh. Permanent checkpoints
#       every 5B so any 5B point (incl. step5960 = 25B) is a brancheable/upcycle-able checkpoint.
#       50B = 11,921 steps at 4,194,304 tokens/step; fixed_steps = round(k*5e9/4,194,304), k=1..9.
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/stdmoe_32exp_50b_wsd_lr2e-3.sh
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
num_experts=32

warmup_steps=2000
decay_steps=1   # pure stable trunk: flat at peak LR; only the final step touches 0. No real anneal.

runname="stdmoe_32exp_50b_wsd_lr2e-3"

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
