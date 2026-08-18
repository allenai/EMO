# PARENT: "scripts/modular_extension/k32cpt_stage.sh" (structure) +
#         "scripts/meta_learning/emo128_vanilla_100b.sh" (trunk-matched flags).
# DESCRIPTION:
#     - ONE STAGE of the meta_learning phase-2 sequential k=32 CPT sweep: continue
#       pretraining a (32 standard + 1 shared)-expert subset checkpoint on ONE cluster's
#       token shards, with every non-expert parameter frozen (attention, embeddings,
#       norms, lm_head, router). Invoked per cluster by run_k32cpt_arm.sh, which does the
#       expert-subset surgery (extract before / writeback after) around this training job.
#     - STANDARD 32-expert training, no EMO random-pool objective: the pool is pinned to
#       the full subset (min=max=eval=33) because the meta arms (e.g. sametok_ws_lam05)
#       were never trained to support a variety of expert pools. LB/z losses stay at
#       trunk values; with the router frozen their gradients reach experts only
#       indirectly through the hidden states.
#     - Optimizer arm is CARRY only: Adam moments sliced from the arm's 20B checkpoint
#       pool, flat lr 2e-3 from step 0 (warmup 0, decay 1).
#     - 4 nodes per stage (32-expert subset model; smaller than the 128e phase-1 runs).
#       Reduce-dp LB stats therefore run over a 4-node DP world -- consistent across all
#       stages and both arms.
#     - Required env vars (set by the driver): MODEL (vanilla|sametok_ws_lam05|...),
#       CLUSTER (0-31), SUBSET_DIR (extract output), SAVE_DIR, TOKENS (max_duration),
#       plus MODE=beaker.
#
#   MODE=beaker MODEL=sametok_ws_lam05 CLUSTER=0 SUBSET_DIR=... SAVE_DIR=... TOKENS=... \
#       bash scripts/meta_learning/k32cpt_stage.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="meta_learning"
DATA_ROOT="s3://ai2-llm"   # unused (explicit paths below) but launch_common expects it

BEAKER_NODES=4
BEAKER_GPUS=8
BEAKER_PRIORITY=urgent
BEAKER_NO_FOLLOW=1

: "${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
: "${CLUSTER:?set CLUSTER=0..31}"
: "${SUBSET_DIR:?set SUBSET_DIR (extract output containing model_and_optim)}"
: "${SAVE_DIR:?set SAVE_DIR (stage save folder)}"
: "${TOKENS:?set TOKENS (stage token budget)}"

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
CLUSTER_TAG=$(printf 'c%02d' "$CLUSTER")
DATA_GLOB="${WEKA_ROOT}/meta_learning/data/meta128_20B-40B/k32_cpt_tokens_${MODEL}/cluster$(printf '%02d' "$CLUSTER")/train/part-*.npy"

# --- trunk-matched model/objective flags (see emo128_vanilla_100b.sh), subset-sized ---
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=33              # 32 standard + 1 shared (shared = last slot)
min_document_expert_pool=33 # pinned full pool: standard training, no random-pool sampling
max_document_expert_pool=33
eval_document_expert_pool=33
warmup_steps=0              # carry arm: Adam moments carried; flat 2e-3 from step 0
decay_steps=1

runname="k32cpt_${MODEL}_${CLUSTER_TAG}"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${SAVE_DIR}" \
		--load_path="${SUBSET_DIR}/model_and_optim" \
		--load_trainer_state=false \
		--load_optim_state=true \
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
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, k32cpt, ${MODEL}]" \
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
