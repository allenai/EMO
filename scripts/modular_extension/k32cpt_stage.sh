# PARENT: "scripts/modular_extension/emo64_100b130b_baseline.sh" (training flags) +
#         "scripts/models_routerfixed/probe_aux_zonly.sh" (--load_path + freeze pattern).
# DESCRIPTION:
#     - ONE STAGE of the sequential k=32 CPT sweep: continue pretraining a
#       (32 standard + 1 shared)-expert subset checkpoint on ONE cluster's token shards,
#       with every non-expert parameter frozen (attention, embeddings, norms, lm_head,
#       router). Invoked per cluster by run_k32cpt_arm.sh, which does the expert-subset
#       surgery (extract before / writeback after) around this training job.
#     - The subset checkpoint is produced by expert_subset_surgery.py extract from the
#       arm's evolving 64-expert pool; --load_path points at it, trainer state is fresh
#       (custom per-cluster data), optimizer state is loaded (carry arm) or not (fresh
#       arm, which uses a 500-step LR warmup instead).
#     - LB/z losses stay at trunk values ("the EMO loss"); with the router frozen their
#       gradients reach experts only indirectly through the hidden states.
#     - Required env vars (set by the driver): CLUSTER (0-31), ARM (carry|fresh),
#       SUBSET_DIR (extract output), SAVE_DIR, TOKENS (max_duration), plus MODE=beaker.
#
#   MODE=beaker CLUSTER=0 ARM=carry SUBSET_DIR=... SAVE_DIR=... TOKENS=... \
#       bash scripts/modular_extension/k32cpt_stage.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="modular_extension"
DATA_ROOT="s3://ai2-llm"   # unused (explicit paths below) but launch_common expects it

BEAKER_NODES=8              # fixed across stages (reduce-dp LB stats depend on node count)
BEAKER_GPUS=8
BEAKER_PRIORITY=urgent
BEAKER_NO_FOLLOW=1

: "${CLUSTER:?set CLUSTER=0..31}"
: "${ARM:?set ARM=carry|fresh}"
: "${SUBSET_DIR:?set SUBSET_DIR (extract output containing model_and_optim)}"
: "${SAVE_DIR:?set SAVE_DIR (stage save folder)}"
: "${TOKENS:?set TOKENS (stage token budget)}"

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
CLUSTER_TAG=$(printf 'c%02d' "$CLUSTER")
DATA_GLOB="${WEKA_ROOT}/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/k32_cpt_tokens/cluster$(printf '%02d' "$CLUSTER")/train/part-*.npy"

# --- trunk-matched model/objective flags (see emo64_100b130b_baseline.sh), subset-sized ---
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=33              # 32 standard + 1 shared (shared = last slot)
min_document_expert_pool=8
max_document_expert_pool=33
eval_document_expert_pool=33
decay_steps=1

if [ "$ARM" = "fresh" ]; then
    warmup_steps=500        # fresh Adam: short warmup to absorb the moment transient
    load_optim=false
else                        # carry and carry_shuf: Adam moments carried; flat 2e-3 from step 0
    warmup_steps=0
    load_optim=true
fi

runname="k32cpt_${ARM}_${CLUSTER_TAG}"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${SAVE_DIR}" \
		--load_path="${SUBSET_DIR}/model_and_optim" \
		--load_trainer_state=false \
		--load_optim_state=${load_optim} \
		--model.freeze_params='[embeddings.*, lm_head.*, blocks.*.attention.*, blocks.*.attention_norm.*, blocks.*.feed_forward_norm.*, blocks.*.feed_forward_moe.router.*]' \
		--dataset.mix=null \
		--dataset.paths="[${DATA_GLOB}]" \
		--dataset.expand_glob=true \
		--dataset.dtype=uint32 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${TOKENS}, unit: tokens}" \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, k32cpt, ${ARM}]" \
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
