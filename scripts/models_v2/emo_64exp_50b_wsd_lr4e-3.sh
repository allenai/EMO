# PARENT: "scripts/models_sizescaling/emo_1b14b_130b.sh" (EMO architecture) crossed with
#         "scripts/models_v2/stdmoe_64exp_50b_wsd_lr4e-3.sh" (models_v2 WSD-trunk conventions).
# DESCRIPTION:
#     - EMO (model type two-level_lb-batch_reduce-dp_sharedexp_randpool: the published EMO router with
#       two-level batch load-balancing, reduce-over-DP, a shared expert, and random document-expert
#       POOLING) trained as a pure WSD STABLE TRUNK. We take ONLY the EMO architecture + training
#       objective; everything else follows the models_v2 conventions so this is a clean EMO-vs-stdMoE
#       head-to-head against stdmoe_64exp_50b_wsd_lr4e-3:
#         * 64 experts, 1 shared, lb 1e-1, OLMoE-mix-0824, 8 nodes / 64 GPUs.
#         * 50B-token run = 11,921 steps at global_batch_size 1024 * seq 4096 = 4,194,304 tokens/step.
#         * Peak LR 4e-3, WSD with decay_steps=1 (flat at the peak for the whole run; only the single
#           final optimizer step touches 0 -- no baked-in anneal). Decay branches are forked later
#           from any saved checkpoint via launch_wsd_decay.sh.
#         * Permanent checkpoints every 5B tokens (fixed_steps) so any 5B point is a brancheable /
#           upcycle-able stable checkpoint; periodic permanent saves disabled; 2-deep rolling
#           ephemerals for preemption safety.
#           fixed_steps = round(k * 5e9 / 4,194,304) for k=1..9 (5,10,...,45B):
#             5B -> 1192   10B -> 2384   15B -> 3576   20B -> 4768   25B -> 5960
#             30B -> 7153  35B -> 8345   40B -> 9537   45B -> 10729   (50B final step 11921 auto-saved)
#     - EMO-specific knobs (vs stdMoE): generate_doc_lengths=true (the randpool router needs per-doc
#       lengths), flash_2 attention backend, and the document-expert pool range
#       min=8 / max=64 / eval=64 (max = all 64 experts; eval uses the full pool).
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/emo_64exp_50b_wsd_lr4e-3.sh
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

# EMO random document-expert pool range (per user: min=8, max=64, eval=64).
min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64

warmup_steps=2000
decay_steps=1   # pure stable trunk: flat at peak LR; only the final step touches 0. No real anneal.

runname="emo_64exp_50b_wsd_lr4e-3"

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
