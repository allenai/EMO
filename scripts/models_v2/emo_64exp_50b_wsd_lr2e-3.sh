# PARENT: "scripts/models_v2/emo_64exp_50b_wsd_lr4e-3.sh"
# DESCRIPTION:
#     - Identical to emo_64exp_50b_wsd_lr4e-3.sh EXCEPT the peak LR is 2e-3 (vs 4e-3). EMO (model type
#       two-level_lb-batch_reduce-dp_sharedexp_randpool) trained as a pure WSD stable trunk, 64 experts
#       / 1 shared, lb 1e-1, 50B tokens, 8 nodes / 64 GPUs, every-5B fixed checkpoints. Pairs with the
#       4e-3 run to sweep peak LR for EMO at this scale, mirroring the stdmoe_64exp_50b_wsd 4e-3/2e-3
#       pair so EMO and stdMoE are compared at matched LRs.
#     - We take ONLY the EMO architecture + training objective; all run conventions (50B, WSD
#       decay_steps=1, every-5B checkpoints, models_v2 save path, wandb emo-extension) follow models_v2.
#       EMO-specific knobs: generate_doc_lengths=true, flash_2 backend, document-expert pool
#       min=8 / max=64 / eval=64.
#       50B = 11,921 steps at 4,194,304 tokens/step; fixed_steps = round(k*5e9/4,194,304), k=1..9.
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/emo_64exp_50b_wsd_lr2e-3.sh
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

# EMO random document-expert pool range (per user: min=8, max=64, eval=64).
min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64

warmup_steps=2000
decay_steps=1   # pure stable trunk: flat at peak LR; only the final step touches 0. No real anneal.

runname="emo_64exp_50b_wsd_lr2e-3"

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
		--model.block.feed_forward_moe.num_experts=${num_experts} \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
