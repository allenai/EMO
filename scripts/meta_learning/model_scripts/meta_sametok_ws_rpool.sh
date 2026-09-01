# PARENT: "scripts/meta_learning/model_scripts/meta_sametok_ws.sh"
# DESCRIPTION:
#     meta_learning lever arm (v): same-tokens working-set + RANDOMIZED OUTER POOL, 100B tokens.
#     Identical to meta128_sametok_ws_100b EXCEPT outer_pool=random: the outer pass samples
#     per-document pool sizes uniformly in [8, 128] exactly like vanilla EMO (instead of pinning
#     keep-all), so the single outer objective trains restricted-forward operation directly —
#     "a selective pseudo-step should improve the model under whatever pool it runs with."
#     Working-set masking and the meta structure are unchanged; evals unaffected.
#     Rationale: same selective-inference gap motivation as the lambda arm, but keeping a single
#     loss (pure FOMAML) instead of a blended objective.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker bash scripts/meta_learning/model_scripts/meta_sametok_ws_rpool.sh
# BUDGET CUT 2026-08-17: phase-1 shortened from 100B to 20B tokens (step 4768; fixed ckpts at
# 10B/20B). Names carry no token budget (WSD flat trunk; budgets change). Phase 2 = k=32-CPT-style cluster-wise selective CPT on tokens 20B-40B (window
# extraction: scripts/meta_learning/eval_scripts/run_extract_20b_40b.sh).
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

EXPERIMENT_NAME="meta_learning"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8
BEAKER_NO_FOLLOW=1

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=128

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=128

warmup_steps=2000
decay_steps=1

inner_lr=3e-1
inner_grad_clip=10
inner_pool_size=32

runname="meta128_sametok_ws_rpool"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 20_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[2384, 4768]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, same_tokens_ws_rpool]" \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--train_module.meta_mode=same_tokens \
		--train_module.inner_lr=${inner_lr} \
		--train_module.inner_grad_clip=${inner_grad_clip} \
		--train_module.inner_pool_size=${inner_pool_size} \
		--train_module.outer_expert_update=working_set \
		--train_module.outer_pool=random
