# PARENT: "scripts/models_fullextend/emo_1b14b_130b.sh"
# DESCRIPTION:
#     - Ghost-expert variant of the EMO 1B/14B 130B-token randpool recipe.
#
#       Method: in addition to the normal document-level expert pool, each
#       document also gets one (or more) "ghost" experts whose weights are a
#       linear combination of the pooled experts' weights:
#           W_ghost = sum_{i in pool} alpha_i * W_i.
#       The ghost competes for tokens like a real expert during the forward
#       pass, but its weights are NEVER instantiated/initialized as parameters —
#       they are recomputed per document from the existing experts. In backward,
#       autograd routes the ghost's gradient straight back into the constituent
#       experts via the alpha-weighted average (d L/d W_i += alpha_i * d L/d W_ghost).
#       The hope: training under a perpetually-simulated "freshly added expert"
#       leaves the trained model amenable to actually instantiating and adding a
#       new expert later.
#
#     - Implemented purely via router-config fields flipped through dotted CLI
#       overrides (same pattern as extension_finetune_mode), so this reuses the
#       published randpool model-type with no new --model-type or argparse args.
#
#     - NOTE: the ghost_extend_* router/MoE logic is NOT yet wired in. These
#       overrides are the config spec for the method; running this before the
#       implementation lands will fail on unknown config keys.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

num_shared_experts=1 # 1 out of 8 will be shared experts

# --- ghost-expert knobs (proposed) ---
ghost_extend_num=1               # number of ghost experts simulated per document
# Coefficient scheme for the blend W_ghost = sum_i alpha_i * W_i over the doc pool:
#   "usage"   = document-usage-weighted (alpha from the doc-level summed expert probs)
#   "uniform" = equal weights over the whole pool
#   "random"  = uniform average over a random sample of ghost_extend_random_k pool experts
ghost_extend_coeff_mode="usage"
ghost_extend_random_k=8          # sample size when coeff_mode="random"
# Routing of the ghost:
#   "always" = additive head; every token in the doc passes through the ghost (clean signal)
#   "topk"   = ghost competes in the token-level top-k against the real experts
ghost_extend_route="always"
# The ghost's router row is itself the same alpha-blend of the pool's router rows, so backprop
# updates the existing router rows in every coeff mode. detach=true cuts the extra grad path that
# only the "usage" blend adds (alpha depends on the router probs); the blended-router-row path is
# unaffected. Leave false to train the router to be averageable.
ghost_extend_detach_coeff=false

runname="emo_1b14b_130b_ghostexpert"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model.block.feed_forward_moe.num_experts=128 \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--model.block.feed_forward_moe.router.ghost_extend_mode=true \
		--model.block.feed_forward_moe.router.ghost_extend_num=${ghost_extend_num} \
		--model.block.feed_forward_moe.router.ghost_extend_coeff_mode=${ghost_extend_coeff_mode} \
		--model.block.feed_forward_moe.router.ghost_extend_random_k=${ghost_extend_random_k} \
		--model.block.feed_forward_moe.router.ghost_extend_route=${ghost_extend_route} \
		--model.block.feed_forward_moe.router.ghost_extend_detach_coeff=${ghost_extend_detach_coeff}
